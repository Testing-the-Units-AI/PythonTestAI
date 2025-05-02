import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_new_tokenizer', type=bool, default=False, help="Needed if no .model file in project root")
parser.add_argument('--train_new_model', type=bool, default=False)
parser.add_argument('--framework', help="Do you want to generate pytest or unittest tests?")
parser.add_argument('--input_code_file', default=False, help="Where do you want to get code to generate tests for?")
parser.add_argument('--output_test_file', type=bool, default=False, help="Where do you want to output tests?")

args = parser.parse_args()

train_new_tokenizer = getattr(args, "train_new_tokenizer", False)
train_new_model = getattr(args, "train_new_model", False)
framework = getattr(args, "framework", None)
input_code_file = getattr(args, "input_code_file", None)
output_test_file = getattr(args, "output_test_file", None)

print(f"Running script with:")
print(f"  train_new_tokenizer = {train_new_tokenizer}")
print(f"  train_new_model     = {train_new_model}")
print(f"  framework           = {framework}")
print(f"  input_code_file     = {input_code_file}")
print(f"  output_test_file    = {output_test_file}")

# CONSTANTS

DATA_DIR = './data/raw' # Where all the raw stories are
TRAIN_FILE = 'data/all.jsonl'
TEST_FILE = 'data/dataset.jsonl'
TRAIN_TOKENIZER_FILE = 'data/dataset.jsonl'
CORPUS_FILE = 'corpus.txt' # Where all the raw data will be stored
MODEL_WEIGHTS_DIR = "final_model_weights.pth"

TOKENIZER_PREFIX = 'bpe_model' # Tokenizer name
TOKENIZER_PATH = TOKENIZER_PREFIX + ".model"
PAD_TOKEN_ID = 5

VOCAB_SIZE = 7017
MAX_TRAIN_SEQ_LEN = 1024
MAX_GEN_SEQ_LEN = 1024


#MODIFIABLE CONSTANTS FOR MODEL TRAINING START HERE
DEVICE = "cuda"

BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = .002
# Dictates creativity of the model, < 1 more deterministic, > 1 more creative/stochastic, 1 is no change from base model.
TEMPERATURE = .9
EARLY_EPOCH_STOP = 2
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = .2
N_HEADS = 8


# IMPORTANT FUNCTIONS

from FinalProjHelper import *
import torch
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def collate_fn(batch):
    """
    This function takes in all of the batches from the dataset, which will be jagged arrays, and
    inserts padding tokens <pad> such that all of the sequences are the same length. Note that the
    Cross Entropy Loss criterion should specify to ignore the index 3 so it does not affect the training.

    :param batch: The batch of prompts and completions that form a jagged array to be padded.
    :return: The input and label batches properly padded.
    """

    enc_input_batch, dec_input_batch, target_batch = zip(*batch)
    enc_input_batch = nn.utils.rnn.pad_sequence(enc_input_batch, batch_first=True, padding_value=PAD_TOKEN_ID)
    dec_input_batch = nn.utils.rnn.pad_sequence(dec_input_batch, batch_first=True, padding_value=PAD_TOKEN_ID)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=PAD_TOKEN_ID)
    return enc_input_batch, dec_input_batch, target_batch


def train_model(model, device, tokenizer, model_type=""):
    """
    The main training code generalized for all of the models,
    including evaluation metrics such as loss graphs, BLEU score, and Perplexity score.

    :param model: The instantiated model needing training.
    :param device: The device the model should be trained on, preferrably cuda.
    :param tokenizer: The loaded tokenizer trained on the dataset being used for model training.
    :return: The progression of training and testing losses.
    """
    # Loading tokenizer file and getting most up to date vocab size
    vocab_size = tokenizer.get_piece_size()
    print('v size: ', vocab_size)

    # Set up datasets from the given jsonl files for training
    train_data = TextDatasetTED(TRAIN_FILE, tokenizer, MAX_TRAIN_SEQ_LEN)
    test_data = TextDatasetTED(TEST_FILE, tokenizer, MAX_TRAIN_SEQ_LEN)

    # Using pytorch DataLoaders for easy batching, shuffling, etc.
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Adding on a decaying learning rate to the optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    best_test_loss = float('inf')
    no_improve_epochs = 0

    train_losses, test_losses = [], []
    for epoch in range(EPOCHS):
        print('Epoch: ', epoch)

        # Emptying cache and unused data on every epoch since CUDA would run out of memory otherwise
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Want the model in training mode
        model.train()
        total_train_loss = 0

        for enc_input_ids, dec_input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            enc_input_ids = enc_input_ids.to(device)
            dec_input_ids = dec_input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            # Get the padding masks for proper training
            enc_pad_mask = (enc_input_ids == PAD_TOKEN_ID)
            dec_pad_mask = (dec_input_ids == PAD_TOKEN_ID)

            # Getting the probability distributions for the prompts...
            logits = model(enc_input_ids, dec_input_ids, enc_pad_mask, dec_pad_mask)

            """
            For understanding this dimension change, understand that the logits are of
            dimension (B,S,V) (see forward() function of base model for explanantion) and the targets are of
            dimension (B,S) (where each entry is the correct token to predict for that position in the sequence).

            For Cross Entropy Loss, we must have 1 prediction for each row in both tensors. Therefore, if we can reduce
            each tensor such that it reads the last dimension for each row, the function will work. Aka we would have
            dimension (B x S, V) (every row is a token's probability distribution) and
            dimension (B x S) (every entry is that token's correct value, in chronological order with the logits)

            The .view() function allows us to do this my making the last dimension of each tensor account for each entry.
            """
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            # Adjusting weights...
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Don't want the model to train on testing data, so .eval()
        model.eval()
        total_test_loss = 0

        # Evaluate testing loss after training on this epoch to see performance on new data
        with torch.no_grad():
            for enc_input_ids, dec_input_ids, target_ids in test_loader:
                enc_input_ids = enc_input_ids.to(device)
                dec_input_ids = dec_input_ids.to(device)
                target_ids = target_ids.to(device)

                # Get the padding masks for proper training
                enc_pad_mask = (enc_input_ids == PAD_TOKEN_ID)
                dec_pad_mask = (dec_input_ids == PAD_TOKEN_ID)

                # Getting the probability distributions for the prompts...
                logits = model(enc_input_ids, dec_input_ids, enc_pad_mask, dec_pad_mask)

                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            scheduler.step(avg_test_loss)

            # If our testing data starts getting worse over time, we can stop it early to reduce losses in accuracy based on a preset constant
            if (avg_test_loss < best_test_loss):
                best_test_loss = avg_test_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

        if (no_improve_epochs >= EARLY_EPOCH_STOP):
            print(f"No improvement in {EARLY_EPOCH_STOP} epochs, stopping...")
            break

    torch.save(model.state_dict(), MODEL_WEIGHTS_DIR)
    print(f"Saved model weights to {MODEL_WEIGHTS_DIR}")

    print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}")
    print(f"Model Perplexity: {Perplexity(avg_train_loss):.4f} Model BLEU: {BLEU(model, tokenizer, test_loader):.4f}")
    plotLossOverEpochs(len(train_losses), train_losses, test_losses, model_type)

    return train_losses, test_losses


def plotLossOverEpochs(epochs, train_loss, test_loss, model_type=""):
    """
    Creates a plot showing the losses over time for a model.

    :param epochs: The number of epochs the training took place over
    :param train_loss: The losses of training over the epochs
    :param test_loss: The losses of testing over the epochs
    :param name: The name of the trained model being evaluated
    """
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(model_type + " Loss per Epoch")

    x_range = range(1, epochs + 1)

    plt.plot(x_range, train_loss)
    plt.plot(x_range, test_loss)

    plt.plot(x_range, train_loss, label="Training Loss", color='blue')
    plt.plot(x_range, test_loss, label="Testing Loss", color='orange')

    plt.legend()
    plt.show()


# Because the loss is already cross entropy, we can just do the natural exponentiation of the loss
# Loss here is the average loss across tokens
def Perplexity(loss):
    return math.exp(loss)


def BLEU(model, tokenizer, test_loader):
    """
        Evaluates BLEU score of the entire model by getting probabilities.

        :param epochs: The number of epochs the training took place over
        :param train_loss: The losses of training over the epochs
        :param test_loss: The losses of testing over the epochs
        :param name: The name of the trained model being evaluated
        :return: The overall BLEU scoring of the prompts and completions ran on
    """

    model.eval()
    references = []
    candidates = []
    samples_processed = 0
    smoothing_function = SmoothingFunction().method1

    with torch.no_grad():

        # We really just want the raw prompts and completions to get rid of unnecessary padding
        for enc_input_ids, dec_input_ids, target_ids in test_loader:

            enc_input_ids = enc_input_ids.to(DEVICE)
            dec_input_ids = dec_input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            # Get the padding masks for proper training
            enc_pad_mask = (enc_input_ids == PAD_TOKEN_ID)
            dec_pad_mask = (dec_input_ids == PAD_TOKEN_ID)

            # The model does teacher forcing predictions, which is exactly what we need to compare with the labels
            logits = model(enc_input_ids, dec_input_ids, enc_pad_mask, dec_pad_mask)

            # Taking the best token from each probability distribution for comparison against the labels
            predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            target_ids = target_ids.cpu().tolist()

            for predicted, target in zip(predicted_ids, target_ids):

                # Process 250 samples so it doesnt run forever
                if samples_processed > 250:
                    break

                # For each prediction vector and label vector, decode it and add it to a list for BLEU scoring
                pred_decode = tokenizer.decode(predicted)
                reference = tokenizer.decode(target)

                samples_processed += 1

                candidates.append(pred_decode.split())
                references.append([reference.split()])

    # Compute the corpus-level BLEU score. For the purposes of this project, up to 3-gram comparisons were made
    bleu_score = corpus_bleu(references, candidates, weights=(.25, .25, .25, .25),
                             smoothing_function=smoothing_function)
    return bleu_score


def prompt_model(model, tokenizer, name):
    with open(input_code_file, "r") as infile:
        prompt = infile.read().strip()

    generated_test = model.generate(
        tokenizer,
        prompt,
        max_seq_length=MAX_GEN_SEQ_LEN,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        temperature=TEMPERATURE,
        device=DEVICE
    )

    with open(output_test_file, "w") as outfile:
        outfile.write(f"Prompt:\n{prompt}\n")
        outfile.write("Generated Unit Test:\n")
        outfile.write(generated_test + "\n")


# MAIN CODE



tokenizer = Tokenizer(TOKENIZER_PREFIX)
print('expected V size: ', VOCAB_SIZE)
if train_new_tokenizer:
    tokenizer.train(vocab_size=VOCAB_SIZE, jsonl_file=TRAIN_TOKENIZER_FILE, sample_limit=None)

device = torch.device(DEVICE)
tokenizer.load()

# Sanity Check:
ids = tokenizer.encode("def my_func(lalala: str) -> List[int]:\n    print(\"foo bar\")\n    return [3, 1, 4, 1, 5, 9]")
code = tokenizer.decode(ids)
print(code)

print("piece size: ", tokenizer.get_piece_size(), " and vocab size: ", tokenizer.sp.vocab_size())

for i in range(10):
    print(tokenizer.sp.IdToPiece(i))




# TOKENIZER


from FinalProjModels import TransformerEDLanguageModel

transformer_model = TransformerEDLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    enc_num_layers=NUM_LAYERS,
    dec_num_layers=NUM_LAYERS,
    n_heads=N_HEADS,
    dropout=DROPOUT,
    pad_token_id=PAD_TOKEN_ID,
    seq_len=MAX_TRAIN_SEQ_LEN,
    name="Transformer Encoder-Decoder"
).to(device)

if train_new_model:
    train_model(transformer_model, device, tokenizer, transformer_model.name)
else:
    print("Loading old weights...")
    transformer_model.load_state_dict(torch.load(MODEL_WEIGHTS_DIR)
)




# TRANSFORMER ED

prompt_model(transformer_model, tokenizer, transformer_model.name)
