import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=86 --patch-size=256 --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=64 --patch-size=256 --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=215 --patch-size=256 --weight-decay=1e-4 --save-checkpoints',

    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=18 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=6 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=32 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',

    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=9 --patch-size=800 --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=4 --patch-size=800 --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=20 --patch-size=800 --weight-decay=1e-4 --save-checkpoints',    
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
