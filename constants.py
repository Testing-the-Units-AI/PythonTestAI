
DATA_DIR = './data/raw' # Where all the raw stories are
TRAIN_FILE = './data/train.jsonl'
TEST_FILE = './data/test.jsonl'
CORPUS_FILE = 'corpus.txt' # Where all the raw data will be stored

TOKENIZER_PREFIX = 'bpe_tokenizer' # Tokenizer name
TOKENIZER_PATH = TOKENIZER_PREFIX + ".model"
PAD_TOKEN_ID = 3

VOCAB_SIZE = 10000 # Based on project handout, limit of vocab tokens allowed
MAX_TRAIN_SEQ_LEN = 128
MAX_GEN_SEQ_LEN = 50


#MODIFIABLE CONSTANTS FOR MODEL TRAINING START HERE
DEVICE = "cuda"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = .002
# Dictates creativity of the model, < 1 more deterministic, > 1 more creative/stochastic, 1 is no change from base model.
TEMPERATURE = 1.1
EARLY_EPOCH_STOP = 4

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = .2
# Nucleus sampling, 0 picks from most likely token only, >= 1 picks from all tokens. .75-.9 is a good range
TOP_P = .8

N_HEADS = 8 # For the transformer model only
