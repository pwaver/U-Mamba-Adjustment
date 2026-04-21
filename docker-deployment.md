# Docker deployment to AWS (Shape 1: persistent EC2 GPU instance)

Companion to [aws-deployment-setup.md](aws-deployment-setup.md), which covers
the bare-metal / `setup_aws.sh` path. This doc covers the Docker path:

- Image built locally with [Dockerfile](Dockerfile) (via
  [scripts/docker-build.sh](scripts/docker-build.sh))
- Pushed to **Amazon ECR** (via
  [scripts/docker-push-ecr.sh](scripts/docker-push-ecr.sh))
- Pulled and run on a persistent **EC2 g5.xlarge** instance with an Elastic IP
- Dataset lives in **S3**, results sync back to S3 every ~10 min
- Edited remotely via **VS Code Remote-SSH**, usable from anywhere on the road

## Architecture

```
     ┌───────────────────┐  git push           ┌────────────────┐
     │  Home machine     │ ─────────────────▶ │  GitHub        │
     │  (Quadro RTX 4000)│                     │  pwaver/U-...  │
     └────────┬──────────┘                     └────────────────┘
              │
              │ docker build + push              ┌────────────────┐
              ├────────────────────────────────▶ │  AWS ECR       │
              │                                  │  umamba-ts:g5  │
              │                                  └───────┬────────┘
              │ aws s3 sync (one-time)                   │ docker pull
              │                                          ▼
              │                                  ┌────────────────┐
              └────────────────────────────────▶ │  AWS S3        │
                                                 │  dataset, logs │
                                                 └───────┬────────┘
                                                         ▲
                                                         │ mount as /data
                                                         │
              ┌──────────────────────────────────┐       │
              │  EC2 g5.xlarge (A10G, 24 GB)     │ ──────┘
              │  Deep Learning AMI + Elastic IP  │
              │  ┌────────────────────────────┐  │
              │  │ container: umamba-ts:g5    │  │
              │  │  nnUNetv2_train ...        │  │
              │  └────────────────────────────┘  │
              └─────────────────┬────────────────┘
                                │ VS Code Remote-SSH (from any laptop)
                                ▼
                       ┌────────────────┐
                       │  Traveling     │
                       │  laptop        │
                       └────────────────┘
```

## 1. One-time AWS setup

### 1a. IAM user for you (home machine)

Your IAM user needs these AWS-managed policies (IAM console → Users → your
user → Add permissions → Attach policies directly):

- `AmazonEC2ContainerRegistryPowerUser` — push/pull ECR images
- `AmazonS3FullAccess` (or a custom policy scoped to `s3:*` on your bucket)
- `AmazonEC2FullAccess` — launch/stop instances, manage Elastic IPs
- `IAMFullAccess` — create the instance role in step 1b (revoke once done if
  you prefer least-privilege)

### 1b. IAM role for EC2 (so the container can pull + sync without baked creds)

IAM console → Roles → Create role → Trusted entity type = **AWS service**,
Use case = **EC2**. Attach:

- `AmazonEC2ContainerRegistryReadOnly` — docker pull from ECR
- `AmazonS3FullAccess` (or scoped to your dataset bucket) — dataset/results sync

Name it something like `umamba-training-instance-role`. At instance-launch
time you'll select it as the "IAM instance profile."

### 1c. ECR repository

Created automatically by [scripts/docker-push-ecr.sh](scripts/docker-push-ecr.sh)
on first push. If you want to pre-create manually:

```bash
aws ecr create-repository --repository-name umamba-ts --region us-east-1
```

### 1d. S3 bucket

Already done for dataset staging — see
[aws-deployment-setup.md](aws-deployment-setup.md) §5. Use the same bucket for
results sync too, under a different prefix (e.g. `s3://bucket/results/`).

## 2. Build + push the image (from home machine)

One-time or whenever trainer code / deps change:

```bash
# Build for the target GPU arch (8.6 = A10G; see docker-build.sh for others)
TORCH_CUDA_ARCH_LIST="8.6" IMAGE_TAG="umamba-ts:g5" bash scripts/docker-build.sh

# Push to ECR (first push ~12 GB; later pushes only upload changed layers)
bash scripts/docker-push-ecr.sh
```

After the first successful push, remember the remote URI. It's of the form:

```
<aws-account>.dkr.ecr.us-east-1.amazonaws.com/umamba-ts:g5
```

## 3. Launch the EC2 instance

### 3a. Console clicks (first time, ~5 min)

EC2 console → **Launch instances**:

| Field | Value |
| --- | --- |
| Name | `umamba-ts-training` |
| AMI | **Deep Learning AMI GPU PyTorch — Ubuntu 22.04** (has NVIDIA driver + Container Toolkit) |
| Instance type | `g5.xlarge` (cheapest A10G) |
| Key pair | Create new or use existing; save the `.pem` to `~/.ssh/` |
| Network / VPC | default |
| Security group | Create new, inbound SSH (port 22) from **My IP** only |
| Storage | 200 GB gp3 (image is 22 GB, dataset is multi-GB, leave headroom) |
| **Advanced details** → IAM instance profile | `umamba-training-instance-role` (from step 1b) |

Launch. Wait ~60 sec for state = running.

### 3b. Elastic IP (stable DNS across stop/start)

EC2 console → **Elastic IPs** → Allocate → Associate with the instance.

Cost: **free while attached to a running instance**, $0.005/hr (~$3.60/mo)
while the instance is stopped. Worth it so your SSH config doesn't rot.

Note the Elastic IP's public DNS. That's your stable connection point.

### 3c. SSH config (from home laptop + travel laptop)

Add to `~/.ssh/config`:

```
Host umamba-aws
    HostName <elastic-ip-or-public-dns>
    User ubuntu
    IdentityFile ~/.ssh/<your-key>.pem
    ServerAliveInterval 60
```

Test:

```bash
ssh umamba-aws   # should log in without prompting for password
```

## 4. First-time instance bootstrap (on the instance)

```bash
ssh umamba-aws

# Verify GPU + Container Toolkit (DLAMI has both preinstalled)
nvidia-smi                                   # A10G, driver >= 535
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Pull our image (IAM role provides ECR creds automatically)
REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password --region $REGION \
    | docker login --username AWS --password-stdin $REGISTRY
docker pull $REGISTRY/umamba-ts:g5
docker tag $REGISTRY/umamba-ts:g5 umamba-ts:g5   # short local alias

# Stage dataset from S3 to local disk
mkdir -p ~/Angiostore/nnUNet_raw ~/Angiostore/nnUNet_preprocessed ~/Angiostore/nnUNet_results
aws s3 sync s3://angiowave-nnunet-datasets/Dataset430/ \
            ~/Angiostore/nnUNet_raw/Dataset430_Angiography3d/
```

## 5. End-to-end training sanity check

```bash
# Quick GPU-visibility check inside the container (on the instance)
docker run --rm --gpus all umamba-ts:g5 python -c "\
import torch; \
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
from mamba_ssm import Mamba3; \
print('Mamba3 ok')"

# Preprocess (writes nnUNet_preprocessed/ on host, ~5 min)
docker run --rm --gpus all \
    -v ~/Angiostore/nnUNet_raw:/data/raw \
    -v ~/Angiostore/nnUNet_preprocessed:/data/preprocessed \
    -v ~/Angiostore/nnUNet_results:/data/results \
    umamba-ts:g5 \
    nnUNetv2_plan_and_preprocess -d 430 --verify_dataset_integrity

# Train (long-running — use screen/tmux or detach)
docker run -d --gpus all \
    --name umamba-train \
    -v ~/Angiostore/nnUNet_raw:/data/raw \
    -v ~/Angiostore/nnUNet_preprocessed:/data/preprocessed \
    -v ~/Angiostore/nnUNet_results:/data/results \
    -e RESULTS_S3_URI=s3://angiowave-nnunet-datasets/results/ \
    -e RESULTS_SYNC_INTERVAL=600 \
    umamba-ts:g5 \
    nnUNetv2_train 430 3d_fullres 0 -tr nnUNetTrainerUMambaTSBot3

# Watch logs
docker logs -f umamba-train
```

Checkpoints land in `~/Angiostore/nnUNet_results/` on the instance and are
rsynced to `s3://.../results/` every 10 min by the entrypoint's background
loop (the `RESULTS_S3_URI` + `RESULTS_SYNC_INTERVAL` env vars enable it).

## 6. Stop / resume workflow (for travel)

### Stopping (end of session)

```bash
# From your laptop:
aws ec2 stop-instances --instance-ids <instance-id>
```

You pay only for EBS (~$20/mo for 200 GB gp3) + Elastic IP (~$3.60/mo)
while stopped. No GPU charge.

### Resuming (from any location)

```bash
aws ec2 start-instances --instance-ids <instance-id>
# wait ~60 sec
ssh umamba-aws
docker start umamba-train   # resume the training container (if it was stopped)
docker logs -f umamba-train
```

The Elastic IP means `umamba-aws` in your SSH config Just Works across
stop/start cycles.

## 7. VS Code Remote-SSH (edit code as if local)

From any laptop with VS Code + "Remote - SSH" extension installed:

1. F1 → "Remote-SSH: Connect to Host" → `umamba-aws`
2. Open folder: `/home/ubuntu/U-Mamba-Adjustment` (clone it on the instance
   first if not there: `git clone https://github.com/pwaver/U-Mamba-Adjustment.git`)
3. Terminal opens on the instance; file edits, git, extensions all run remote.

To edit code inside the **running container** (rare — usually edit on host and
rebuild the image), install the "Dev Containers" extension and attach to the
`umamba-train` container.

## 8. Updating the image

When you change trainer code, rebuild + push. Layer caching means only the
small layers change — push is seconds to minutes, not the full 12 GB.

```bash
# Home machine
TORCH_CUDA_ARCH_LIST="8.6" IMAGE_TAG="umamba-ts:g5" bash scripts/docker-build.sh
bash scripts/docker-push-ecr.sh

# Instance
docker pull $REGISTRY/umamba-ts:g5
docker tag $REGISTRY/umamba-ts:g5 umamba-ts:g5
docker stop umamba-train && docker rm umamba-train
# ... then re-run the `docker run -d` command from §5
```

## 9. Cost summary

| Item | Cost | Notes |
| --- | --- | --- |
| g5.xlarge, running | ~$1.00/hr | on-demand; spot saves ~65% but can be interrupted |
| g5.xlarge, stopped | $0 | (EBS + EIP still billed) |
| 200 GB gp3 EBS | ~$20/mo | persists across stop/start |
| Elastic IP, attached to running instance | $0 | |
| Elastic IP, attached to stopped instance | ~$3.60/mo | |
| ECR storage | $0.10/GB/mo | ~$2/mo for a 22 GB image, less compressed |
| S3 storage (dataset + results) | $0.023/GB/mo | |
| S3 egress (only if you pull data back to laptop) | $0.09/GB | |

Rough monthly floor with the instance stopped: **~$25** (EBS + EIP + ECR + S3).
Actual bill scales with how many hours you let the GPU run.

## 10. Troubleshooting

### `docker pull` fails with "no basic auth credentials"
ECR login expires after 12 hours. Re-run:
```bash
aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin <registry>
```

### Training container exits immediately
Check `docker logs umamba-train`. Most common: `cuda_available=False` because
you forgot `--gpus all`.

### `ModuleNotFoundError` inside container
Rebuild the image — the host `.venv` has no effect on what's inside the
container. All deps are baked at build time.

### Instance disk fills up
Model checkpoints + preprocessing caches accumulate. Either grow the EBS
volume (EC2 console → volumes → Modify, then `sudo growpart` + `resize2fs`)
or clear old results: `rm -rf ~/Angiostore/nnUNet_results/Dataset430_*/old_fold`.

### Spot interruption during training
Not applicable with on-demand. If you switch to spot: set
`RESULTS_S3_URI` + `RESULTS_SYNC_INTERVAL=60` so the latest checkpoint is in
S3; nnUNetv2 can resume from the most recent epoch by adding `--c` to the
train command.
