import datetime
import enum
import pathlib
import shutil
import subprocess

training_setups = [
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=200 --batch-size=180 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=200 --batch-size=100 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=200 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=200 --batch-size=33 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=200 --batch-size=15 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=44 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b4" --epochs=200 --batch-size=14 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="efficientnet-b7" --epochs=200 --batch-size=8 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="MAnet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=16 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=186 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=62 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=272 --patch-size=256 --use-amp --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=28 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=11 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=62 --patch-size=640 --use-amp --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b4" --epochs=200 --batch-size=18 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="efficientnet-b7" --epochs=200 --batch-size=7 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',
    # r'./.venv/Scripts/python train.py --model="DeepLabV3Plus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=34 --patch-size=800 --use-amp --weight-decay=1e-4 --save-checkpoints',

    # r'./.venv/Scripts/python train.py --model="FPN" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp'

    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --weight-decay=1e-3',
    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-6 --weight-decay=1e-3', # astral-serenity-412
    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-6 --weight-decay=1e-4', # fearless-capybara-413
    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-6 --weight-decay=1e-5', # dulcet-flower-415
    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-6 --weight-decay=1e-3',
    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-6 --weight-decay=1e-4',

    r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-5 --weight-decay=1e-4 --cool-down-epochs=100 --dropout=0.3',
    # r'python train_1x1conv.py --model-idx=2 --epochs=400 --learning-rate=1e-6 --weight-decay=1e-3 --cool-down-epochs=60 --dropout=0.5',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
