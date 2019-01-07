from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8 #16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 35
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 56
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

print("config.TRAIN.decay_every:", config.TRAIN.decay_every)

## train set location
config.TRAIN.hr_img_path = '../train_data_out_2' 
config.TRAIN.lr_img_path = '../train_data_in' 


config.VALID = edict()
## test set location
config.VALID.hr_img_path = '../test_data_out_2/' 
config.VALID.lr_img_path = '../test_data_in/' 

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
