import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-3 --dropout=0.1 --cuda 0',
    r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-4 --dropout=0.1 --cuda 0',
    r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=115 --weight-decay=1e-3 --dropout=0.1 --cuda 0',
    r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-4 --cool-down-epochs=115 --weight-decay=1e-5 --dropout=0.1 --cuda 0',
    # r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=100 --dropout=0.1 --cuda 0 --weight-decay=1e-4',
    # r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-3 --cool-down-epochs=100 --dropout=0.15 --cuda 0 --weight-decay=1e-3',
    # r'python train_1x1conv.py --class-idx=0 --model-idx=0 --epochs=400 --learning-rate=1e-4 --cool-down-epochs=100 --dropout=0.15 --cuda 0 --weight-decay=1e-4',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
