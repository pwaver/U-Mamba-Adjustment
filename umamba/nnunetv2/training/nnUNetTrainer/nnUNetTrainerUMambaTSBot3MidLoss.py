import torch
from torch import autocast, nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaTSBot3 import nnUNetTrainerUMambaTSBot3
from nnunetv2.utilities.helpers import dummy_context


class _MiddleSliceLoss3D(nn.Module):
    """Compute the wrapped loss only on the central depth slice.

    For a 3D patch (B, C, D, H, W), drops dim 2 at index D//2 on both
    prediction and target, leaving (B, C, H, W) for the inner loss.
    The network still produces all D output slices — only the middle
    slice's gradient flows back. Temporal context in the encoder /
    decoder Mamba scans is fully preserved.
    """
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, output, target):
        d_o = output.shape[2] // 2
        d_t = target.shape[2] // 2
        return self.loss(output[:, :, d_o], target[:, :, d_t])


class nnUNetTrainerUMambaTSBot3MidLoss(nnUNetTrainerUMambaTSBot3):
    """nnUNetTrainerUMambaTSBot3 with loss restricted to the middle z-slice."""

    def _build_loss(self):
        loss = super()._build_loss()
        # DeepSupervisionWrapper.loss is the per-scale inner loss; replacing it
        # makes every DS scale evaluate only its middle slice without changing
        # the deep-supervision weighting.
        if self.enable_deep_supervision:
            loss.loss = _MiddleSliceLoss3D(loss.loss)
        else:
            loss = _MiddleSliceLoss3D(loss)
        return loss

    def validation_step(self, batch: dict) -> dict:
        # Mirrors nnUNetTrainer.validation_step but reduces output/target to
        # the middle z-slice before the online "fake Dice" tally, so the
        # validation curve reflects the slice the loss is actually training.
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        d_o = output.shape[2] // 2
        d_t = target.shape[2] // 2
        output = output[:, :, d_o]
        target = target[:, :, d_t]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
