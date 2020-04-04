from config import config_local as config
import glob
import os

# checkpoint_path = config.CHECKPOINT_PATH
list_of_checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_PATH, '*.params'))      # /home/milton/Documents/Projects/Trial_results/car_mxsqueezenet_local_output/checkpoints/SqueezeNet-0005.params
checkpoint_epoch_number = [file.split("/")[-1] for file in list_of_checkpoint_files] # ['SqueezeNet-0005.params', 'SqueezeNet-0002.params', ..]
checkpoint_epoch_number = [file.split(".")[0] for file in checkpoint_epoch_number]   # ['SqueezeNet-0005', 'SqueezeNet-0002', ..]
checkpoint_epoch_number = max([int(file.split("-")[-1]) for file in checkpoint_epoch_number])  # [0005, 0004]  [5, 4, 3, 2]
print(checkpoint_epoch_number)
