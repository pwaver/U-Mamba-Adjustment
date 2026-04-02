# nnUNet Training Command Line Calls

## UMambaBot3 (Mamba2 bottleneck, 2D)

```bash
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaBot3
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaBot3_LowDoseContrastSim
```

## UMambaEnc3 (Mamba2 encoder, 2D)

```bash
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaEnc3
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaEnc3_LowDoseContrastSim
```

## UMambaTSBot (temporospatial bottleneck: 2D falls back to UMambaBot, 3D uses temporal-first scan)

```bash
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaTSBot
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaTSBot_LowDoseContrastSim
```

## UMambaTSEnc (temporospatial encoder: 2D falls back to UMambaEnc, 3D uses temporal-first scan)

```bash
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaTSEnc
nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaTSEnc_LowDoseContrastSim
```

---

*Replace `332` with the dataset ID and `2d` with the configuration as needed.*
