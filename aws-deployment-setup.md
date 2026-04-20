# AWS Deployment Setup

Step-by-step: take a freshly-launched AWS GPU instance from zero to
`nnUNetv2_train` running `UMambaTSBot3_3d` against the 3-D angiography
dataset.

The local Ubuntu x86 dev machine (Quadro RTX 4000) already runs this stack;
this document reproduces it on a remote AWS GPU. The heavy lifting is in
[scripts/setup_aws.sh](scripts/setup_aws.sh) — this doc explains the
surrounding workflow.

---

## 1. Pick and launch the AWS instance

| Instance | GPU | VRAM | `TORCH_CUDA_ARCH_LIST` | Notes |
| --- | --- | --- | --- | --- |
| `g5.xlarge` | A10G | 24 GB | `8.6` | Cheapest viable for 3-D training |
| `g5.2xlarge` – `g5.12xlarge` | A10G ×1–4 | 24 GB each | `8.6` | More CPU / multi-GPU |
| `g6.xlarge` | L4 | 24 GB | `8.9` | Newer, Ada Lovelace |
| `p4d.24xlarge` | A100 ×8 | 40 GB each | `8.0` | Large training |
| `p5.48xlarge` | H100 ×8 | 80 GB each | `9.0` | Fastest |

Pick an AMI with recent NVIDIA drivers pre-installed (≥535). **Deep Learning
AMI GPU PyTorch (Ubuntu 22.04)** is a safe default — it skips driver hassle.
If you instead pick stock Ubuntu, install the driver manually before step 3:

```bash
sudo apt update && sudo apt install -y nvidia-driver-535
sudo reboot
```

Reboot is required the first time a driver is installed.

Root disk: allocate **≥ 100 GB** (the combined torch+CUDA userspace wheels,
micromamba env, build caches, and dataset easily exceed 40 GB).

---

## 2. Clone the repo

```bash
ssh ubuntu@<instance-dns>

sudo apt update && sudo apt install -y git build-essential
git clone <your-git-remote>/U-Mamba-Adjustment.git
cd U-Mamba-Adjustment
```

Make sure the clone is complete and the GPU is visible:

```bash
nvidia-smi        # driver + GPU listed
git rev-parse HEAD
```

---

## 3. Run the provisioning script

```bash
bash scripts/setup_aws.sh
```

What it does, in order:

1. Installs **uv** (Python package manager) if not present.
2. Installs **micromamba** (standalone, ~17 MB) to `~/.local/bin/micromamba`.
3. Creates `~/micromamba/envs/cuda128` with **CUDA 12.8 toolchain**
   (`cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`) — just the compiler and
   headers, no Python.
4. Auto-detects the GPU compute capability via `nvidia-smi` and sets
   `TORCH_CUDA_ARCH_LIST` accordingly.
5. Runs `uv sync --extra gpu --no-build-isolation`, which:
   - Materializes the project `.venv` from `pyproject.toml` + `uv.lock`
     (torch, nnunetv2 editable, SimpleITK, etc.).
   - Compiles **causal-conv1d 1.6.1** (~1 min).
   - Compiles **mamba-ssm** from the pinned git commit
     (`316ed6036538405f767782132f76caf342256d33` → provides `Mamba3`,
     which the PyPI 2.3.1 release does not) (~15 min).
6. Runs a smoke test: imports `Mamba3`, constructs `MambaLayer3(dim=320)`,
   runs a forward on a random `(1, 320, 5, 8, 8)` tensor. Fails loudly if
   the kernels didn't compile / load.
7. Writes a `.envrc` file at the repo root so future shells can activate
   the full environment with a single `source .envrc`.

Override the auto-detected GPU arch if needed:
```bash
TORCH_CUDA_ARCH_LIST="9.0" bash scripts/setup_aws.sh
```

Tune parallelism for smaller instances:
```bash
MAX_JOBS=2 bash scripts/setup_aws.sh
```

**Expected wall-clock on a g5.xlarge:** ~20–30 min (dominated by
mamba-ssm CUDA kernel compile). On a p4d/p5 with more vCPUs and
`MAX_JOBS=8`, closer to 10 min.

### Disk footprint after setup

| Location | Size |
| --- | --- |
| `.venv/` | ~6 GB (torch + CUDA userspace wheels) |
| `~/micromamba/envs/cuda128/` | ~3 GB (nvcc + headers) |
| `~/.cache/uv/` | ~2 GB (wheel cache) |

---

## 4. Activate the environment in future shells

```bash
cd U-Mamba-Adjustment
source .envrc
```

`.envrc` sets `CUDA_HOME`, prepends `$CUDA_HOME/bin` to `PATH`, adds
`libcudart.so` to `LD_LIBRARY_PATH`, and activates `.venv`. After this:

```bash
nvcc --version                           # 12.8.93
python -c "import torch; print(torch.cuda.is_available())"   # True
which nnUNetv2_train                     # .venv/bin/nnUNetv2_train
```

*(Optional but recommended: install **direnv** — `sudo apt install direnv`
then `direnv allow` inside the repo — and the environment loads
automatically when you `cd` in.)*

---

## 5. Transfer the prepared dataset

The [prepare_data_for_nnunetv2_3d.py](umamba/nnunetv2/BB-explore/prepare_data_for_nnunetv2_3d.py)
script writes to `~/Angiostore/nnUnet_raw/Dataset430_Angiography3d/`. Mirror
that directory to the AWS box. Three options, in rough order of
preference for large datasets:

```bash
# (a) S3 round-trip — best for multi-GB datasets, reproducible
aws s3 sync ~/Angiostore/nnUnet_raw/Dataset430_Angiography3d \
            s3://<your-bucket>/Dataset430_Angiography3d/
# then on AWS:
aws s3 sync s3://<your-bucket>/Dataset430_Angiography3d/ \
            ~/Angiostore/nnUnet_raw/Dataset430_Angiography3d/

# (b) rsync over ssh — simpler but slow over WAN
rsync -aP --info=progress2 ~/Angiostore/nnUnet_raw/Dataset430_Angiography3d \
      ubuntu@<instance-dns>:~/Angiostore/nnUnet_raw/

# (c) Re-run the prep script on AWS if the source HDF5 files are already
#     available there (e.g., from EFS or S3-mounted)
.venv/bin/python umamba/nnunetv2/BB-explore/prepare_data_for_nnunetv2_3d.py
```

Also mirror or regenerate the preprocessed plans directory if not running
`nnUNetv2_plan_and_preprocess` fresh on AWS:

```bash
# usually just regenerate on AWS (small, quick):
export nnUNet_raw=~/Angiostore/nnUnet_raw
export nnUNet_preprocessed=~/Angiostore/nnUnet_preprocessed
export nnUNet_results=~/Angiostore/nnUnet_results
mkdir -p $nnUNet_preprocessed $nnUNet_results
nnUNetv2_plan_and_preprocess -d 430 --verify_dataset_integrity
```

---

## 6. Train

```bash
source .envrc     # if not already active
export nnUNet_raw=~/Angiostore/nnUnet_raw
export nnUNet_preprocessed=~/Angiostore/nnUnet_preprocessed
export nnUNet_results=~/Angiostore/nnUnet_results

# 3D UMambaTSBot3 training on fold 0:
nnUNetv2_train 430 3d_fullres 0 -tr nnUNetTrainerUMambaTSBot3
```

Training progress and checkpoints land in
`~/Angiostore/nnUnet_results/Dataset430_Angiography3d/…`. Consider
periodically `aws s3 sync`-ing that directory back to S3 so results
survive the instance being stopped.

---

## 7. Teardown / cost control

Stop vs terminate:

- **Stop** the instance when idle — root EBS is retained, you pay only
  for EBS storage (~$0.10/GB/month for gp3). Restarting picks up where
  you left off; no re-run of `setup_aws.sh` needed.
- **Terminate** if done for weeks — detach the EBS volume first (or
  snapshot it) if you want to preserve the env without paying for the
  full volume.

Spot instances cut GPU cost ~70 % but can be interrupted — only use
them if your training writes checkpoints frequently and `nnUNetv2` can
resume from the latest one.

---

## Troubleshooting

### `nvcc: command not found` in a fresh shell
`.envrc` wasn't sourced. `cd` to the repo root and `source .envrc`.

### `ImportError: libcudart.so.12: cannot open shared object file`
Same cause — `LD_LIBRARY_PATH` not set. Source `.envrc`.

### Build fails with `fatal error: cuda_runtime_api.h: No such file`
The build did not find the conda-forge headers. Verify:
```bash
ls ~/micromamba/envs/cuda128/targets/x86_64-linux/include/cuda_runtime_api.h
```
If missing, re-run `bash scripts/setup_aws.sh` — the micromamba env
creation probably failed silently.

### `Mamba3` import fails after setup
Two common causes: (a) CUDA arch mismatch — the kernel was compiled for a
different GPU. Rerun with `TORCH_CUDA_ARCH_LIST=<your-arch>
bash scripts/setup_aws.sh` after deleting `.venv`. (b) torch version
drift — `uv sync` resolved a torch that's ABI-incompatible with the
pre-built kernels. Force rebuild: `uv pip install --no-build-isolation
--force-reinstall causal-conv1d mamba-ssm` with the setup-script env
vars exported.

### Out-of-memory during training
Reduce `batch_size` in
`data/nnUNet_preprocessed/Dataset430_Angiography3d/nnUNetPlans.json`
under the `3d_fullres` configuration, or use a larger instance (see
table in §1).

### Driver ↔ CUDA version mismatch
`nvidia-smi` shows the driver's *max supported* CUDA, not what's
installed. As long as `nvidia-smi` reports CUDA ≥ 12.8, the
micromamba toolchain works. If the driver is older (≤ 525), upgrade
it before running the script.
