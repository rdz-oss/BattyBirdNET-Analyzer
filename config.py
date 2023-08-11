#################
# Misc settings #
#################

# Random seed for gaussian noise
RANDOM_SEED = 42

##########################
# Model paths and config #
##########################
# These BirdNET  models are necessary also for detecting bats as we use their embeddings and classify
# them to identify the bats.
# MODEL_PATH = 'checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model' # This will load the protobuf model
MODEL_PATH = 'checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'
MDATA_MODEL_PATH = 'checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite'
LABELS_FILE = 'checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt'
TRANSLATED_LABELS_PATH = 'labels/V2.4'
TRANSLATED_BAT_LABELS_PATH = 'labels/bats/'

# Path to custom trained classifier
# If None, no custom classifier will be used
# Make sure to set the LABELS_FILE above accordingly
CUSTOM_CLASSIFIER = None

##################
# Audio settings #
##################

# BirdNET uses a sample rate of 48kHz, so the model input size is
# (batch size, 48000 kHz * 3 seconds) = (1, 144000)
# Recordings will be resampled automatically.
# For bats we use: 144000 for 1 sec.
# Note that only SIG_LENGTH * SAMPLING_RATE = 144000 combinations will work,
# values possible e.g. 144000 240000 360000 check your classifier frequency!
SAMPLE_RATE: int = 144000

# We're using 1-second chunks
SIG_LENGTH: float = 144000 / SAMPLE_RATE

# Define overlap between consecutive chunks < SIG_LENGTH; 0 = no overlap
SIG_OVERLAP: float = SIG_LENGTH / 4.0

# Define minimum length of audio chunk for prediction, 
# chunks shorter than SIG_LENGTH seconds will be padded with zeros
SIG_MINLEN: float = SIG_LENGTH / 3.0

#####################
# Metadata settings #
#####################
# These settings are currently not in use for bat detection
LATITUDE = -1
LONGITUDE = -1
WEEK = -1
LOCATION_FILTER_THRESHOLD = 0.03

######################
# Inference settings #
######################

# If None or empty file, no custom species list will be used
# Note: Entries in this list have to match entries from the LABELS_FILE
# We use the 2021 eBird taxonomy for species names (Clements list)
CODES_FILE = 'eBird_taxonomy_codes_2021E.json'
SPECIES_LIST_FILE = 'example/species_list.txt'

# File input path and output path for selection tables
INPUT_PATH: str = 'example/'
OUTPUT_PATH: str = 'example/'

# Used for bats - the files here are supposed to be analyzed by default setting
INPUT_PATH_SAMPLES: str = 'put-your-files-here/'
OUTPUT_PATH_SAMPLES: str = 'put-your-files-here/results/'
BAT_CLASSIFIER_LOCATION: str = 'checkpoints/bats/v1.0'

ALLOWED_FILETYPES = ['wav', 'flac', 'mp3', 'ogg', 'm4a']

# Number of threads to use for inference.
# Can be as high as number of CPUs in your system
CPU_THREADS: int = 8
TFLITE_THREADS: int = 6

# False will output logits, True will convert to sigmoid activations
APPLY_SIGMOID: bool = True
SIGMOID_SENSITIVITY: float = 1.0

# Minimum confidence score to include in selection table 
# (be aware: if APPLY_SIGMOID = False, this no longer represents 
# probabilities and needs to be adjusted)
MIN_CONFIDENCE: float = 0.6

# Number of samples to process at the same time. Higher values can increase
# processing speed, but will also increase memory usage.
# Might only be useful for GPU inference.
BATCH_SIZE: int = 1

# Specifies the output format. 'table' denotes a Raven selection table,
# 'audacity' denotes a TXT file with the same format as Audacity timeline labels
# 'csv' denotes a CSV file with start, end, species and confidence.
RESULT_TYPE = 'csv'

#####################
# Training settings #
#####################

# Training data path
TRAIN_DATA_PATH = 'train_data/'

# Number of epochs to train for
TRAIN_EPOCHS: int = 100

# Batch size for training
TRAIN_BATCH_SIZE: int = 32

# Learning rate for training
TRAIN_LEARNING_RATE: float = 0.01

# Number of hidden units in custom classifier
# If >0, a two-layer classifier will be trained
TRAIN_HIDDEN_UNITS: int = 0

#####################
# Misc runtime vars #
#####################
CODES = {}
LABELS: list[str] = []
TRANSLATED_LABELS: list[str] = []
SPECIES_LIST: list[str] = []
ERROR_LOG_FILE: str = 'error_log.txt'
FILE_LIST = []
FILE_STORAGE_PATH = ''


######################
# Get and set config #
######################

def get_config():
    return {
        'RANDOM_SEED': RANDOM_SEED,
        'MODEL_PATH': MODEL_PATH,
        'MDATA_MODEL_PATH': MDATA_MODEL_PATH,
        'LABELS_FILE': LABELS_FILE,
        'CUSTOM_CLASSIFIER': CUSTOM_CLASSIFIER,
        'SAMPLE_RATE': SAMPLE_RATE,
        'SIG_LENGTH': SIG_LENGTH,
        'SIG_OVERLAP': SIG_OVERLAP,
        'SIG_MINLEN': SIG_MINLEN,
        'LATITUDE': LATITUDE,
        'LONGITUDE': LONGITUDE,
        'WEEK': WEEK,
        'LOCATION_FILTER_THRESHOLD': LOCATION_FILTER_THRESHOLD,
        'CODES_FILE': CODES_FILE,
        'SPECIES_LIST_FILE': SPECIES_LIST_FILE,
        'INPUT_PATH': INPUT_PATH,
        'OUTPUT_PATH': OUTPUT_PATH,
        'CPU_THREADS': CPU_THREADS,
        'TFLITE_THREADS': TFLITE_THREADS,
        'APPLY_SIGMOID': APPLY_SIGMOID,
        'SIGMOID_SENSITIVITY': SIGMOID_SENSITIVITY,
        'MIN_CONFIDENCE': MIN_CONFIDENCE,
        'BATCH_SIZE': BATCH_SIZE,
        'RESULT_TYPE': RESULT_TYPE,
        'TRAIN_DATA_PATH': TRAIN_DATA_PATH,
        'TRAIN_EPOCHS': TRAIN_EPOCHS,
        'TRAIN_BATCH_SIZE': TRAIN_BATCH_SIZE,
        'TRAIN_LEARNING_RATE': TRAIN_LEARNING_RATE,
        'TRAIN_HIDDEN_UNITS': TRAIN_HIDDEN_UNITS,
        'CODES': CODES,
        'LABELS': LABELS,
        'TRANSLATED_LABELS': TRANSLATED_LABELS,
        'SPECIES_LIST': SPECIES_LIST,
        'ERROR_LOG_FILE': ERROR_LOG_FILE,
        'INPUT_PATH_SAMPLES': INPUT_PATH_SAMPLES,
        'OUTPUT_PATH_SAMPLES': OUTPUT_PATH_SAMPLES,
        'BAT_CLASSIFIER_LOCATION': BAT_CLASSIFIER_LOCATION,
        'TRANSLATED_BAT_LABELS_PATH': TRANSLATED_BAT_LABELS_PATH
    }


def set_config(c):
    global RANDOM_SEED
    global MODEL_PATH
    global MDATA_MODEL_PATH
    global LABELS_FILE
    global CUSTOM_CLASSIFIER
    global SAMPLE_RATE
    global SIG_LENGTH
    global SIG_OVERLAP
    global SIG_MINLEN
    global LATITUDE
    global LONGITUDE
    global WEEK
    global LOCATION_FILTER_THRESHOLD
    global CODES_FILE
    global SPECIES_LIST_FILE
    global INPUT_PATH
    global OUTPUT_PATH
    global CPU_THREADS
    global TFLITE_THREADS
    global APPLY_SIGMOID
    global SIGMOID_SENSITIVITY
    global MIN_CONFIDENCE
    global BATCH_SIZE
    global RESULT_TYPE
    global TRAIN_DATA_PATH
    global TRAIN_EPOCHS
    global TRAIN_BATCH_SIZE
    global TRAIN_LEARNING_RATE
    global TRAIN_HIDDEN_UNITS
    global CODES
    global LABELS
    global TRANSLATED_LABELS
    global SPECIES_LIST
    global ERROR_LOG_FILE
    global INPUT_PATH_SAMPLES
    global OUTPUT_PATH_SAMPLES
    global BAT_CLASSIFIER_LOCATION
    global TRANSLATED_BAT_LABELS_PATH

    RANDOM_SEED = c['RANDOM_SEED']
    MODEL_PATH = c['MODEL_PATH']
    MDATA_MODEL_PATH = c['MDATA_MODEL_PATH']
    LABELS_FILE = c['LABELS_FILE']
    CUSTOM_CLASSIFIER = c['CUSTOM_CLASSIFIER']
    SAMPLE_RATE = c['SAMPLE_RATE']
    SIG_LENGTH = c['SIG_LENGTH']
    SIG_OVERLAP = c['SIG_OVERLAP']
    SIG_MINLEN = c['SIG_MINLEN']
    LATITUDE = c['LATITUDE']
    LONGITUDE = c['LONGITUDE']
    WEEK = c['WEEK']
    LOCATION_FILTER_THRESHOLD = c['LOCATION_FILTER_THRESHOLD']
    CODES_FILE = c['CODES_FILE']
    SPECIES_LIST_FILE = c['SPECIES_LIST_FILE']
    INPUT_PATH = c['INPUT_PATH']
    OUTPUT_PATH = c['OUTPUT_PATH']
    CPU_THREADS = c['CPU_THREADS']
    TFLITE_THREADS = c['TFLITE_THREADS']
    APPLY_SIGMOID = c['APPLY_SIGMOID']
    SIGMOID_SENSITIVITY = c['SIGMOID_SENSITIVITY']
    MIN_CONFIDENCE = c['MIN_CONFIDENCE']
    BATCH_SIZE = c['BATCH_SIZE']
    RESULT_TYPE = c['RESULT_TYPE']
    TRAIN_DATA_PATH = c['TRAIN_DATA_PATH']
    TRAIN_EPOCHS = c['TRAIN_EPOCHS']
    TRAIN_BATCH_SIZE = c['TRAIN_BATCH_SIZE']
    TRAIN_LEARNING_RATE = c['TRAIN_LEARNING_RATE']
    TRAIN_HIDDEN_UNITS = c['TRAIN_HIDDEN_UNITS']
    CODES = c['CODES']
    LABELS = c['LABELS']
    TRANSLATED_LABELS = c['TRANSLATED_LABELS']
    SPECIES_LIST = c['SPECIES_LIST']
    ERROR_LOG_FILE = c['ERROR_LOG_FILE']
    INPUT_PATH_SAMPLES = c['INPUT_PATH_SAMPLES']
    OUTPUT_PATH_SAMPLES = c['OUTPUT_PATH_SAMPLES']
    BAT_CLASSIFIER_LOCATION = c['BAT_CLASSIFIER_LOCATION']
    TRANSLATED_BAT_LABELS_PATH = c['TRANSLATED_BAT_LABELS_PATH']
