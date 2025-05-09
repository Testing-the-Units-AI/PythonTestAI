import torch

TRAIN_FILE = 'data/all.jsonl'
TEST_FILE = 'data/dataset.jsonl'
TRAIN_TOKENIZER_FILE = 'data/dataset.jsonl'

TOKENIZER_PREFIX = 'test'
TOKENIZER_PATH = "./TokenizerModels/" + TOKENIZER_PREFIX + ".model"
PAD_TOKEN_ID = 5

VOCAB_SIZE = 7017
MAX_GEN_SEQ_LEN = 1024

#MODIFIABLE CONSTANTS FOR MODEL TRAINING START HERE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = .002
# Dictates creativity of the model, < 1 more deterministic, > 1 more creative/stochastic, 1 is no change from base model.
TEMPERATURE = .9
TOP_K = 4
EARLY_EPOCH_STOP = 2
EPOCHS_PER_SAVE = 1
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = .2
N_HEADS = 8
MAX_TRAIN_SEQ_LEN = 1024

TRAINING_SAVE_DIR = "./TrainingSaves"

# Constants for model prompting
MODEL_INPUT_DIR = "./ModelInputCode"
MODEL_OUTPUT_DIR = "./ModelOutputUnitTests"