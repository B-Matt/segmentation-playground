import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=300 --batch-size=22 --patch-size=800 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=300 --batch-size=12 --patch-size=800 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=300 --batch-size=8 --patch-size=800 --save-checkpoints',

    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=300 --batch-size=36 --patch-size=640 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=2 --batch-size=22 --patch-size=640 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=300 --batch-size=10 --patch-size=640 --save-checkpoints',

    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=300 --batch-size=10 --patch-size=256 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=300 --batch-size=10 --patch-size=256 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=300 --batch-size=10 --patch-size=256 --save-checkpoints',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
