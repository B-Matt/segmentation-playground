import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=300 --batch-size=216 --patch-size=256 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=300 --batch-size=112 --patch-size=256 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=300 --batch-size=256 --patch-size=256 --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=300 --batch-size=36 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=300 --batch-size=12 --patch-size=640 --weight-decay=1e-4 --use-amp=false --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=300 --batch-size=44 --patch-size=640 --weight-decay=1e-4 --save-checkpoints',

    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=300 --batch-size=20 --patch-size=800 --weight-decay=1e-4 --use-amp=false --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=300 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --use-amp=false --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=300 --batch-size=6 --patch-size=800 --weight-decay=1e-4 --use-amp=false --save-checkpoints',    
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
