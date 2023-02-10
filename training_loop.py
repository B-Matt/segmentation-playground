import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=3 --batch-size=180 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=3 --batch-size=100 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=200 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=200 --batch-size=16 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=200 --batch-size=8 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=24 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=200 --batch-size=11 --patch-size=800 --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=200 --batch-size=5 --patch-size=800 --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=16 --patch-size=800 --weight-decay=1e-4 --save-checkpoints',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
