from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1

_C.TARGET_TRAIN = True          ##use for source free DA and select train source or train target
_C.SOURCE_MODEL_DIR = ''        ##use for source free DA and when train target give source model path

# Directory to save the output files
_C.OUTPUT_DIR = './output_temp'
# Path to a directory where the files were saved
_C.RESUME = ''
# Set seed to negative value to random everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.USE_CUDA = True
# Print detailed information (e.g. what trainer,
# dataset, backbone, etc.)
_C.VERBOSE = True

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (256, 256)
_C.INPUT.CROP_SIZE = (224, 224)
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()
_C.INPUT.TRANSFORMS_TEST = ()
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Default mean and std come from ImageNet
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Padding for random crop
_C.INPUT.CROP_PADDING = 4
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.
_C.INPUT.GN_STD = 0.15
# RandomAugment
_C.INPUT.RANDAUGMENT_N = 2
_C.INPUT.RANDAUGMENT_M = 10

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = './datasets'
_C.DATASET.NAME = ''
# List of names of source domains
_C.DATASET.SOURCE_DOMAINS = ()
# List of names of target domains
_C.DATASET.TARGET_DOMAINS = ()
# # Number of labeled instances for the SSL setting
# _C.DATASET.NUM_LABELED = 250
# Percentage of validation data (only used for SSL datasets)
# Set to 0 if do not want to use val data
# Using val data for hyperparameter tuning was done in Oliver et al. 2018
_C.DATASET.VAL_PERCENT = 0.1
# Fold index for STL-10 dataset (normal range is 0 - 9)
# Negative number means None
_C.DATASET.STL10_FOLD = -1

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
# Apply transformations to an image K times (during training)
_C.DATALOADER.K_TRANSFORMS = 1
# Setting for train_s data-loader
_C.DATALOADER.TRAIN_S = CN()
_C.DATALOADER.TRAIN_S.SAMPLER = 'RandomSampler'
_C.DATALOADER.TRAIN_S.BATCH_SIZE = 32

_C.DATALOADER.TRAIN_S.VAL_SOURCE = False
# Parameter for RandomDomainSampler
# 0 or -1 means sampling from all domains
_C.DATALOADER.TRAIN_S.N_DOMAIN = 0

# Setting for train_t data-loader
_C.DATALOADER.TRAIN_T = CN()
# Set to false if you want to have unique
# data loader params for train_t
_C.DATALOADER.TRAIN_T.SAME_AS_S = True
_C.DATALOADER.TRAIN_T.SAMPLER = 'RandomSampler'
_C.DATALOADER.TRAIN_T.BATCH_SIZE = 32
_C.DATALOADER.TRAIN_T.N_DOMAIN = 0

# Setting for test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = 'SequentialSampler'
_C.DATALOADER.TEST.BATCH_SIZE = 32

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights for initialization
_C.MODEL.INIT_WEIGHTS = ''
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ''
_C.MODEL.BACKBONE.PRETRAINED = True

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = 'adam'
_C.OPTIM.LR = 0.0003
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.99
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = 'single_step'
_C.OPTIM.STEPSIZE = (10, )
_C.OPTIM.GAMMA = 0.1

### for UDA or Multi-source DA
_C.OPTIM.MAX_EPOCH = 10

###for source free DA
_C.OPTIM.SOURCE_MAX_EPOCH = 10
_C.OPTIM.TARGET_MAX_EPOCH = 10

###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to disable
_C.TRAIN.CHECKPOINT_FREQ = 1
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
# Use 'train_s', 'train_t' , 'smaller_one' or 'bigger_one' to count
# the number of iterations in an epoch
_C.TRAIN.COUNT_ITER = 'bigger_one'

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = 'Classification'
_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# How often (epoch) to do testing during training
# Set to 0 or negative value to disable
_C.TEST.EVAL_FREQ = 1
# Use 'test' set or 'val' set for evaluation
_C.TEST.SPLIT = 'test'

###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = ''
# MSDTR
_C.TRAINER.MSDTR = CN()
_C.TRAINER.MSDTR.HIDDEN = ()
_C.TRAINER.MSDTR.DIS_SIZE = 500
_C.TRAINER.MSDTR.RECONSTRUCTION = 1
_C.TRAINER.MSDTR.cls_d = 0.1
_C.TRAINER.MSDTR.adv_begin = 0
_C.TRAINER.MSDTR.mask_b = 0.6
_C.TRAINER.MSDTR.mask_k = 3             ####(note that mask_b + 1/mask_k<1)
_C.TRAINER.MSDTR.g = 0.1
_C.TRAINER.MSDTR.lr_muti_g = 0.0
_C.TRAINER.MSDTR.lr_muti_ddi = 0.0
_C.TRAINER.MSDTR.lr_muti_cdi = 0.0
_C.TRAINER.MSDTR.lr_muti_dds = 0.0
_C.TRAINER.MSDTR.lr_muti_cds = 0.0
_C.TRAINER.MSDTR.lr_muti_r = 0.0
_C.TRAINER.MSDTR.lr_muti_dis = 0.0





