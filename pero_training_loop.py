import datetime
import subprocess

training_setups = [
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b4" --epochs=200 --batch-size=140 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b7" --epochs=200 --batch-size=70 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=70 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b4" --epochs=200 --batch-size=22 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b7" --epochs=200 --batch-size=12 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b4" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b7" --epochs=200 --batch-size=6 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',

    r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b4" --epochs=200 --batch-size=155 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b7" --epochs=200 --batch-size=95 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="resnext50_32x4d" --epochs=200 --batch-size=316 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b4" --epochs=200 --batch-size=25 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b7" --epochs=200 --batch-size=12 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="resnext50_32x4d" --epochs=200 --batch-size=40 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b4" --epochs=200 --batch-size=12 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b7" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    r'./venv/Scripts/python train.py --model="FPN" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
