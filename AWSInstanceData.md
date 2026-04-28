# AWSInstanceData

AMI: ami-00613b158c7a09b63
Instance: i-062e3992e2d61ba72
Public IP: 34.203.34.66

410847265439.dkr.ecr.us-east-1.amazonaws.com/umamba-ts:g5 
vpc-00c800bb4ab240fd1

aws ecr describe-images   --repository-name umamba-ts   --image-ids imageTag=g5   --region us-east-1
{
    "imageDetails": [
        {
            "registryId": "410847265439",
            "repositoryName": "umamba-ts",
            "imageDigest": "sha256:ae571fca892288f21adbcc0008f3f22d910610a7438c78103fd61d3f03b49cd5",
            "imageTags": [
                "g5"
            ],
            "imageSizeInBytes": 12300617456,
            "imagePushedAt": "2026-04-20T23:58:01.143000-04:00",
            "imageManifestMediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "artifactMediaType": "application/vnd.docker.container.image.v1+json",
            "lastRecordedPullTime": "2026-04-24T21:58:25.242000-04:00"
        }
    ]
}
--ipc=host (or --shm-size) is required for any PyTorch workload with num_workers > 0 in a container — it's not specific to nnU-Net or your image.

ssh -i ~/.ssh/umamba-train.pem ubuntu@34.203.34.66 \
  'nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader'

## Whole file

ssh -i ~/.ssh/umamba-train.pem ubuntu@34.203.34.66 \
  'cat /home/ubuntu/nnunet/results/Dataset430_Angio/nnUNetTrainerUMambaTSBot3__nnUNetPlans__3d_fullres/fold_0/training_log_2026_4_25_04_51_45.txt'

## Last 50 lines (usually what you actually want — recent epochs, current state)

ssh -i ~/.ssh/umamba-train.pem ubuntu@34.203.34.66 \
  'tail -50 /home/ubuntu/nnunet/results/Dataset430_Angio/nnUNetTrainerUMambaTSBot3MidLoss__nnUNetPlans__3d_fullres/fold_0/training_log_2026_4_25_23_17_59.txt'

/home/ubuntu/nnunet/results/Dataset430_Angio/nnUNetTrainerUMambaTSBot3MidLoss__nnUNetPlans__3d_fullres/fold_0/training_log_2026_4_25_23_17_59.txt

## Follow with

ssh -i ~/.ssh/umamba-train.pem ubuntu@34.203.34.66   'tail -1000 /home/ubuntu/nnunet/results/Dataset430_Angio/nnUNetTrainerUMambaTSBot3MidLoss__nnUNetPlans__3d_fullres/fold_0/training_log_2026_4_25_23_17_59.txt' | awk '/: Epoch [0-9]+/{if(y)printf "%s",b; b=""; y=0} {b=b $0 ORS} /Yayy/{y=1} END{if(y)printf "%s",b}' | tail -10