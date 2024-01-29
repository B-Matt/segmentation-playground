# Custom Semantic Segmentation Trainer

## train.py
The trainer file is utilized for initiating the training process for models. The trainer uses the *segmentation_models_pytorch* library for the models. The execution of the script is facilitated through the utilization of command-line options. The trainer employs the AdamW optimizer, Albumentations library for data augumentation, OneCycleLR learning rate scheduler, Binary Cross Entropy Logits Loss (or Cross Entropy Loss if you want to train more than 2 classes), Automatic Mixed Precision, and an early stopping mechanism inspired by YOLO, with a patience value of 10. The metrics collected, including the dice score (for training and validation), IoU score (for training and validation), loss (for training and validation), and learning rate, are uploaded to WANDB along with the corresponding images. In order to mitigate the occurrence of gradient explosions, we have incorporated gradient clipping, setting the threshold at a value of 255.0. Additionally, the number of epochs in which the trainer does not check for the best checkpoint can be defined by adjusting the *check_best_cooldown* variable in the UnetTraining class.

**Command-line options:**
- *--model* - Which model you want to train *(Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus)
- *--lr* - Maximum learning rate that learning rate scheduler must achieve
- *--adam-eps* - A precision threshold that essentially represents the smallest scaled differential between two floating-point representable numbers in computations
- *--weight-decay* - Decoupled weight decay to apply in AdamW optimizer
- *--batch-size* - Training batch size
- *--encoder* - Backbone encoder
- *--epochs* - Number of training epochs
- *--workers* - Number of DataLoader workers
- *--classes* - Number of classes
- *--patch-size* - Patch size of the training images
- *--pin-memory* - If you want to use pin memory for DataLoader
- *--eval-step* - Run evaluation every # steps
- *--load-model* - Path to the trained model
- *--save-checkpoints* - If you want to save checkpoints after every epoch
- *--use-amp* - If you want to use Automatci Mixed Percision
- *--search-files* - If you want to automatically search directories for the training data (if you have splitted data into two folders (masks & images)). By default it expects: training_dataset.txt file with directory names of images that are in training dataset, validation_dataset.txt file with directory names of images that are in validation dataset, and test_dataset.txt file with directory names of images that are in test dataset.

**Example of training U-Net++ using the ResNeXt model as the backbone, with a maximum of 300 epochs, a batch size of 8, a patch size of 928, AMP, and checkpoints saved after each epoch:**
*python train.py --model="UnetPlusPlus" --encoder="resnext101_32x8d" --epochs=300 --batch-size=8 --patch-size=928 --weight-decay=1e-3 --save-checkpoints --use-amp*

### dataset.py
Consists of two dataset classes that are decoupled from our model training code for better readability and modularity. *Dataset* class is used if user wants to train more than two classes, for our usecase we used *BinaryDataset* class. BinaryDataset class inside constructor preloads all images defined inside .txt file or inside imgs/masks directories. Everytime DataLoader loads new image it first calls load_sample() method that loads both image and mask in memory, resizes & pads it, and make data augumentation if augumentation dictionary is provided in constructor. Every mask is normalized into 0-1 space in order to achieve faster convergence. 


## detect.py
The file used for the inference of trained Semantic Segmentation models. The execution of the script is facilitated by using command-line options. The user has the option to select the model to be used for inference, as well as the patch size and confidence threshold (with a default value of 0.5). If the user utilizes an image or a directory of images as a source, the Detector will automatically load provided images into memory and perform inference on them. If the user provides an RTSP stream or video file as a source, the Detector will load each frame into memory and perform inference. After the inference process, the Detector has the capability to display the inferred image as an OpenCV window. Additionally, the window provides information regarding the inference time and the mean area, measured in pixels, of the largest segmentation polygon present in the image. If the video is too long, you can specify the maximum number of frames on which you want to perform the inference.

**Command-line options:**
- *--model* - Path to the trained model to be used in the inference process
- *--patch-size* - Patch size of the inference
- *--classes* - Number of classes in the model
- *--conf-thres* - Confidence Threshold
- *--source* - Image/video file/RTSP URL/directory with image/video files
- *--encoder* - Encoder of trained model
- *--max-frames* - Max. number of processed frames
- *--view-img* - Show inference result (if you input a video or stream, each frame will be returned as a result)
- *--save-video* - If you inputed video you can save it with inference polygons and metrics

**Example of the inference:**
*python detect.py --model "checkpoints/spring-deluge-294/best-checkpoint.pth.tar" --patch-size 800 --conf-thres 0.5 --encoder "resnext50_32x4d" --source "1512411018435.jpg" --view-img*
*python detect.py --model "checkpoints/matej-data-sc/best-checkpoint.pth.tar" --patch-size 640 --conf-thres 0.5 --encoder "mit_b4" --source rtsp://192.168.0.4/defaultPrimary1?streamType=u --view-img*
*python detect.py --model "checkpoints/spring-deluge-294/best-checkpoint.pth.tar" --patch-size 800 --conf-thres 0.5 --encoder "resnext50_32x4d" --source "playground/examples/wood.mp4"*

### dataloaders.py
Consists of the *LoadImages* (used for loading images and videos) and *LoadStreams* (used for loading RTSP streams) classes, which are used for loading images or frames of videos/rtsp streams as iterables. On each iteration, these classes will load image/frame, resize and pad it, and perform data argumentation (if required). They will then return the image/frame's path, the augumented image, the loaded image, the OpenCV capture instance, the current frame, and the total number of frames.
