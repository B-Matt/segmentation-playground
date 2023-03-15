import datetime
import enum
import pathlib
import shutil
import subprocess

# class-idx=1  -> FFMNet2
training_setups = [
    r'python train_1x1conv.py --class-idx=1 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-5 --dropout=0.1 --cuda 1 --start-reduced-dimension 2 --end-reduced-dimension 64 --middle-reduced-dimension 8',
    r'python train_1x1conv.py --class-idx=1 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-5 --dropout=0.1 --cuda 1 --start-reduced-dimension 2 --end-reduced-dimension 40 --middle-reduced-dimension 16',
    r'python train_1x1conv.py --class-idx=1 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-5 --dropout=0.1 --cuda 1 --start-reduced-dimension 2 --end-reduced-dimension 50 --middle-reduced-dimension 32',
    r'python train_1x1conv.py --class-idx=1 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-5 --dropout=0.1 --cuda 1 --start-reduced-dimension 3 --end-reduced-dimension 64 --middle-reduced-dimension 8',
    r'python train_1x1conv.py --class-idx=1 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-5 --dropout=0.1 --cuda 1 --start-reduced-dimension 3 --end-reduced-dimension 40 --middle-reduced-dimension 16',
    r'python train_1x1conv.py --class-idx=1 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-5 --dropout=0.1 --cuda 1 --start-reduced-dimension 3 --end-reduced-dimension 50 --middle-reduced-dimension 32',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
