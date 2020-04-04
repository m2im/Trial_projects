# car_mxsqueezenet_spot_config

# import the necessary packages
from os import path

# define the base path to the cars dataset
BASE_PATH = "/home/ubuntu/dltraining/datasets/cars"

# based on the base path, derive the images path and meta file path
IMAGES_PATH = path.sep.join([BASE_PATH, "car_ims"])
LABELS_PATH = path.sep.join([BASE_PATH, "complete_dataset.csv"])

# define the path to the output training, validation, and testing
# lists
MX_OUTPUT = BASE_PATH
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/test.lst"])

# define the path to the output training, validation, and testing
# image records
TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/test.rec"])

# define the path to the label encoder, checkpoints, and logs
OUTPUT_PATH = "/home/ubuntu/dltraining/output"
LABEL_ENCODER_PATH = path.sep.join([OUTPUT_PATH, "le.cpickle"])
CHECKPOINT_PATH = path.sep.join([OUTPUT_PATH, "checkpoints"])
LOG_PATH = path.sep.join([OUTPUT_PATH, "logs"])


# define the RGB means from the ImageNet dataset
R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939

# define the percentage of validation and testing images relative
# to the number of training images
NUM_CLASSES = 164
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# define the batch size and other network parameters
BATCH_SIZE = 32
NUM_DEVICES = 1
DATA_SHAPE = (3, 227, 227)
MODEL_PREFIX = "SqueezeNet"

# define training and optimizer parameters
START_EPOCH = 0
NUM_EPOCH = 40
LR = 1e-4
WD = 0.0005   # Weight Decay

# define callback and metrics parameters
TOP_K = 5
FREQUENCY = 100   # Speedometer callback