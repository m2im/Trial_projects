# USAGE
# python train_squeezenet_spot.py --start-epoch 0

# import the necessary packages
from config import config_local as config
from pyimagesearch.nn.mxconv import MxSqueezeNet
from imutils import paths
import mxnet as mx
import logging
import json
import glob
import os


# Define the number of epoch where we want to train, or where the spot instance needs to resume training
list_of_checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_PATH, '*.params'))      # /home/milton/Documents/Projects/Trial_results/car_mxsqueezenet_local_output/checkpoints/SqueezeNet-0005.params
checkpoint_epoch_number = [file.split("/")[-1] for file in list_of_checkpoint_files] # ['SqueezeNet-0005.params', 'SqueezeNet-0002.params', ..]
checkpoint_epoch_number = [file.split(".")[0] for file in checkpoint_epoch_number]   # ['SqueezeNet-0005', 'SqueezeNet-0002', ..]
num_of_checkpoint_files = max([int(file.split("-")[-1]) for file in checkpoint_epoch_number])  # [0005, 0004]  [5, 4, 3, 2]  [5]

if config.START_EPOCH < num_of_checkpoint_files:
	START_EPOCH = START_EPOCH
else:
	START_EPOCH = num_of_checkpoint_files

# set the logging level and output file
log_file = "training_{}.log".format(START_EPOCH)
logging.basicConfig(level=logging.DEBUG,
	filename=os.path.sep.join([config.LOG_PATH, log_file]),
	filemode="w")


# load the RGB means for the training set, then determine the batch
# size
batchSize = config.BATCH_SIZE * config.NUM_DEVICES

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
	path_imgrec=config.TRAIN_MX_REC,
	data_shape=config.DATA_SHAPE,
	batch_size=batchSize,
	rand_crop=True,
	rand_mirror=True,
	rotate=15,
	max_shear_ratio=0.1,
	mean_r=config.R_MEAN,
	mean_g=config.G_MEAN,
	mean_b=config.B_MEAN,
	preprocess_threads=config.NUM_DEVICES * 2)

# construct the validation image iterator
valIter = mx.io.ImageRecordIter(
	path_imgrec=config.VAL_MX_REC,
	data_shape=config.DATA_SHAPE,
	batch_size=batchSize,
	mean_r=config.R_MEAN,
	mean_g=config.G_MEAN,
	mean_b=config.B_MEAN)

# initialize the optimizer
opt = mx.optimizer.SGD(learning_rate=config.LR, momentum=0.9, wd=config.WD,
	rescale_grad=1.0 / batchSize)

# construct the checkpoints path, initialize the model argument and
# auxiliary parameters
checkpointsPath = os.path.sep.join([config.CHECKPOINT_PATH,
	config.MODEL_PREFIX])
argParams = None
auxParams = None

# if there is no specific model starting epoch supplied, then
# initialize the network
if START_EPOCH <= 0:
	# build the LeNet architecture
	print("[INFO] building network...")
	model = MxSqueezeNet.build(config.NUM_CLASSES)

# otherwise, a specific checkpoint was supplied
else:
	# load the checkpoint from disk
	print("[INFO] loading epoch {}...".format(START_EPOCH))
	model = mx.model.FeedForward.load(checkpointsPath,
		START_EPOCH)

	# update the model and parameters
	argParams = model.arg_params
	auxParams = model.aux_params
	model = model.symbol

# compile the model
model = mx.model.FeedForward(
	ctx=[mx.cpu(0)],  # ctx=[mx.gpu(0), mx.gpu(1), mx.gpu(2)]
	symbol=model,
	initializer=mx.initializer.Xavier(),
	arg_params=argParams,
	aux_params=auxParams,
	optimizer=opt,
	num_epoch=config.NUM_EPOCH,
	begin_epoch=START_EPOCH)

# initialize the callbacks and evaluation metrics
batchEndCBs = [mx.callback.Speedometer(batchSize, config.FREQUENCY)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=config.TOP_K),
	mx.metric.CrossEntropy()]

# train the network
print("[INFO] training network...")
model.fit(
	X=trainIter,
	eval_data=valIter,
	eval_metric=metrics,
	batch_end_callback=batchEndCBs,
	epoch_end_callback=epochEndCBs)