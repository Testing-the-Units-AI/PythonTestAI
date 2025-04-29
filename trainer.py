import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sentencepiece as spm
import math
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataloader import TextDataset

from constants import *

#
# Load the pre-trained tokenizer from the .model tokenizer file
def load_tokenizer(tokenizer_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer


def collate_fn(batch):
    """
    This function takes in all of the batches from the dataset, which will be jagged arrays, and
    inserts padding tokens <pad> such that all of the sequences are the same length. Note that the
    Cross Entropy Loss criterion should specify to ignore the index 3 so it does not affect the training.

    :param batch: The batch of prompts that form a jagged array to be padded.
    :return: The input and label batches properly padded.
    """

    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=PAD_TOKEN_ID)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=PAD_TOKEN_ID)
    return input_batch, target_batch

def train_model(model, device, tokenizer, model_type="transformer"):
    """ 
    The main training code generalized for all of the models,
    including evaluation metrics such as loss graphs, BLEU score, and Perplexity score.

    :param model: The instantiated model needing training.
    :param device: The device the model should be trained on, preferrably cuda.
    :param tokenizer: The loaded tokenizer trained on the dataset being used for model training.
    :return: The progression of training and testing losses.
    """
    torch.cuda.empty_cache()
    # Loading tokenizer file and getting most up to date vocab size
    vocab_size = tokenizer.get_piece_size()

    # Set up datasets from the given jsonl files for training
    train_data = TextDataset(TRAIN_FILE, tokenizer, MAX_TRAIN_SEQ_LEN)
    test_data = TextDataset(TEST_FILE, tokenizer, MAX_TRAIN_SEQ_LEN)

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
        # Emptying cache and unused data on every epoch since CUDA would run out of memory otherwise
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Want the model in training mode
        model.train()
        total_train_loss = 0

        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            # Getting the probability distributions for the prompts...
            logits, _ = model(input_ids)

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
            for input_ids, target_ids in test_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits, _ = model(input_ids)

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
            break

    print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}")
    print(f"Model Perplexity: {Perplexity(avg_train_loss):.4f} Model BLEU: {BLEU(model, tokenizer, test_loader):.4f}")
    plotLossOverEpochs(EPOCHS, train_losses, test_losses, model_type)

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
        for input_ids, target_ids in test_loader:

            input_ids = input_ids.to(DEVICE)
            # The model does teacher forcing predictions, which is exactly what we need to compare with the labels
            logits, _ = model(input_ids)
            # Taking the best token from each probability distribution for comparison against the labels
            predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            target_ids = target_ids.cpu().tolist()

            for predicted, target in zip(predicted_ids, target_ids):

                # Process 250 samples so it doesnt run forever
                if samples_processed > 250:
                    break

                # For each prediction vector and label vector, decode it and add it to a list for BLEU scoring
                pred_decode = tokenizer.decode(predicted, out_type=str)
                reference = tokenizer.decode(target, out_type=str)

                samples_processed += 1

                candidates.append(pred_decode.split())
                references.append([reference.split()])

    # Compute the corpus-level BLEU score. For the purposes of this project, up to 3-gram comparisons were made
    bleu_score = corpus_bleu(references, candidates, weights=(.33, .34, .33, 0), smoothing_function=smoothing_function)
    return bleu_score