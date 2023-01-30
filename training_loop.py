import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=186 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=62 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=272 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',

    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=28 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=11 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=62 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',

    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=18 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=7 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',
    r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=34 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',    
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
