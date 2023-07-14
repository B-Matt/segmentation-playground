import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-3 --dropout=0.1 --cuda 0',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
