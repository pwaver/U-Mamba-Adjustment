import sys
from nnunetv2.run.run_training import run_training_entry

if __name__ == "__main__":
    # Pass the command line arguments directly to the run_training_entry function
    sys.argv = ["debug_nnunetv2_train.py", "330", "2d", "1", "-tr", "nnUNetTrainer"]
    run_training_entry()