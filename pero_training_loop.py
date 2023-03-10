import datetime
import subprocess

training_setups = [
    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b4" --epochs=200 --batch-size=140 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b7" --epochs=200 --batch-size=70 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=70 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',

    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b4" --epochs=200 --batch-size=22 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b7" --epochs=200 --batch-size=12 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',

    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b4" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="efficientnet-b7" --epochs=200 --batch-size=6 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="UnetPlusPlus" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    #
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b4" --epochs=200 --batch-size=155 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b7" --epochs=200 --batch-size=95 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="resnext50_32x4d" --epochs=200 --batch-size=316 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b4" --epochs=200 --batch-size=25 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b7" --epochs=200 --batch-size=12 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="resnext50_32x4d" --epochs=200 --batch-size=40 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b4" --epochs=200 --batch-size=12 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="efficientnet-b7" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="FPN" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',

    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="efficientnet-b4" --epochs=200 --batch-size=140 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="efficientnet-b7" --epochs=200 --batch-size=70 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=70 --patch-size=256 --weight-decay=1e-4 --save-checkpoints --use-amp',

    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="efficientnet-b4" --epochs=200 --batch-size=22 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="efficientnet-b7" --epochs=200 --batch-size=12 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=640 --weight-decay=1e-4 --save-checkpoints --use-amp',

    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="efficientnet-b4" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="efficientnet-b7" --epochs=200 --batch-size=6 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',
    # r'./venv/Scripts/python train.py --model="PSPNet" --encoder="resnext50_32x4d" --epochs=200 --batch-size=10 --patch-size=800 --weight-decay=1e-4 --save-checkpoints --use-amp',

    # r'./venv/Scripts/python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-5 --weight-decay=1e-4 --cool-down-epochs=100 --dropout=0.3',
    # r'./venv/Scripts/python train_1x1conv.py --model-idx=1 --epochs=400 --learning-rate=1e-6 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.15'

    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-5 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.15',
    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-3 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.15',
    # r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-4 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.15',
    r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-4 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.1',
    r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-4 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.1',
    r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-3 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.1',
    r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-3 --weight-decay=1e-3 --cool-down-epochs=100 --dropout=0.1',
    r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-3 --weight-decay=1e-2 --cool-down-epochs=100 --dropout=0.8',
    r'python train_1x1conv.py --model-idx=0 --epochs=400 --learning-rate=1e-4 --weight-decay=1e-2 --cool-down-epochs=100 --dropout=0.8',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")
