#!/usr/bin/env bash
# docker-push-ecr.sh — push a local image to Amazon ECR.
#
# Creates the ECR repo if it doesn't exist, logs docker in to the registry,
# tags the local image, pushes it, and prints the final URI that any
# AWS EC2 instance can `docker pull`.
#
# Usage:
#   bash scripts/docker-push-ecr.sh                         # defaults
#   ECR_REPO_NAME=umamba-ts bash scripts/docker-push-ecr.sh
#   LOCAL_TAG=umamba-ts:g5 REMOTE_TAG=g5 bash scripts/docker-push-ecr.sh
#   AWS_REGION=us-west-2 bash scripts/docker-push-ecr.sh
#
# Requires: aws CLI v2 configured with creds that can call ecr:*
#           (cheapest IAM policy: AmazonEC2ContainerRegistryPowerUser).

set -euo pipefail

LOCAL_TAG="${LOCAL_TAG:-umamba-ts:g5}"
REMOTE_TAG="${REMOTE_TAG:-g5}"
ECR_REPO_NAME="${ECR_REPO_NAME:-umamba-ts}"

# Region: explicit env var > aws configure > us-east-1 fallback
AWS_REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
AWS_REGION="${AWS_REGION:-us-east-1}"

log() { printf '\n\e[1;34m[ecr-push]\e[0m %s\n' "$*"; }

# ── 0. Preflight: image exists locally, aws creds work ───────────────────
if ! docker image inspect "$LOCAL_TAG" >/dev/null 2>&1; then
    echo "ERROR: local image '$LOCAL_TAG' not found. Build it first:" >&2
    echo "       bash scripts/docker-build.sh" >&2
    exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REMOTE_URI="${REGISTRY}/${ECR_REPO_NAME}:${REMOTE_TAG}"

log "Account:    $ACCOUNT_ID"
log "Region:     $AWS_REGION"
log "Repo:       $ECR_REPO_NAME"
log "Local tag:  $LOCAL_TAG"
log "Remote URI: $REMOTE_URI"

# ── 1. Ensure the ECR repo exists ────────────────────────────────────────
if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" \
       --region "$AWS_REGION" >/dev/null 2>&1; then
    log "ECR repo '$ECR_REPO_NAME' already exists."
else
    log "Creating ECR repo '$ECR_REPO_NAME'..."
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE \
        >/dev/null
fi

# ── 2. Docker login to ECR ────────────────────────────────────────────────
log "Logging docker in to $REGISTRY"
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"

# ── 3. Tag + push ────────────────────────────────────────────────────────
log "Tagging $LOCAL_TAG -> $REMOTE_URI"
docker tag "$LOCAL_TAG" "$REMOTE_URI"

log "Pushing (first push uploads everything, later pushes upload only changed layers)"
docker push "$REMOTE_URI"

log "Done. On any AWS EC2 instance, run:"
cat <<EOF

    aws ecr get-login-password --region $AWS_REGION \\
        | docker login --username AWS --password-stdin $REGISTRY
    docker pull $REMOTE_URI

EOF
