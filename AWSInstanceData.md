AWSInstanceData

AMI: ami-00613b158c7a09b63
Instance: i-062e3992e2d61ba72
Public IP: 34.203.34.66

--ipc=host (or --shm-size) is required for any PyTorch workload with num_workers > 0 in a container — it's not specific to nnU-Net or your image.

ssh -i ~/.ssh/umamba-train.pem ubuntu@34.203.34.66 \
  'nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader'

# Whole file
ssh -i ~/.ssh/umamba-train.pem ubuntu@34.203.34.66 \
  'cat /home/ubuntu/nnunet/results/Dataset430_Angio/nnUNetTrainerUMambaTSBot3__nnUNetPlans__3d_fullres/fold_0/training_log_2026_4_25_04_51_45.txt'

# Last 50 lines (usually what you actually want — recent epochs, current state)
ssh -i ~/.ssh/umamba-train.pem ubuntu@34.203.34.66 \
  'tail -50 /home/ubuntu/nnunet/results/Dataset430_Angio/nnUNetTrainerUMambaTSBot3__nnUNetPlans__3d_fullres/fold_0/training_log_2026_4_25_04_51_45.txt'
