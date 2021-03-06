BUFFER_SIZE = 1024
BATCH_SIZE = 1

# ITERS = 200000
# ITERS_DECAY_START = 100000

ITERS = 100000
START_LR = 2e-4
END_LR = 1e-6
DECAY_POWER = 2



ITERS_INTERVAL_FOR_SUMMARY_LOG = 10
ITERS_INTERVAL_FOR_SAMPLE_GENERATION = 2000
ITERS_INTERVAL_FOR_CHECKPOINT = 2000

IMG_WIDTH = 128
IMG_HEIGHT = 128
OUTPUT_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT,IMG_WIDTH,OUTPUT_CHANNELS)
RESNET_BLOCKS = 6
SKIP = False


LAMBDA_CYC = 10
LAMBDA_ID = LAMBDA_CYC * 0
LAMBDA_LM = 100
LAMBDA_UGD = 0.5
LAMBDA_CGD = 0.5
LAMBDA_LOCAL_D = 0.3

DATASET_DIR = './celeba2bitmoji/'
TRAIN_A_DIR = DATASET_DIR + 'trainA/'
TRAIN_B_DIR = DATASET_DIR + 'trainB/'
TRAIN_A_LM_DIR = DATASET_DIR + 'trainA_lmheatmap/'
TRAIN_B_LM_DIR = DATASET_DIR + 'trainB_lmheatmap/'
TRAIN_A_GENDER_LABELS = DATASET_DIR + 'celeba_genders.npy'
TRAIN_B_GENDER_LABELS = DATASET_DIR + 'bitmoji_genders.npy'
TEST_A_DIR = DATASET_DIR + 'testA/'
TEST_B_DIR = DATASET_DIR + 'testB/'
A_LANDMARK_REGRESSOR_PATH = './landmark_regressor/celeba_lmreg_heat'
B_LANDMARK_REGRESSOR_PATH = './landmark_regressor/bitmoji_lmreg_heat'
ITEMPOOL_SIZE = 50
LOG_DIR = 'logs/'
TRAIN_SAMPLE_LOG_DIR = LOG_DIR + 'train_samples/'
CHECKPOINT_PATH = "checkpoints/train"