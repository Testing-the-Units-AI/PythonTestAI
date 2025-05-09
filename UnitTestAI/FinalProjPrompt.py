import argparse
import time

import torch
import os
from typing import Literal

from FinalProjConstants import MODEL_INPUT_DIR, MODEL_OUTPUT_DIR, MAX_GEN_SEQ_LEN, TOP_K, DEVICE, TRAINING_SAVE_DIR, \
    device, TEMPERATURE, TOKENIZER_PREFIX, VOCAB_SIZE
from FinalProjModels import TestFrameworkType, TransformerEDLanguageModel
from FinalProjHelper import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, Tokenizer

# ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument('--model_input_dir', default=MODEL_INPUT_DIR, help=f"Which directory has the code files for the model input?\n(Defaults to {MODEL_INPUT_DIR}): ")
parser.add_argument('--model_output_dir', default=MODEL_OUTPUT_DIR, help=f"Where do you want to output tests from the model?\n(Defaults to {MODEL_OUTPUT_DIR}): ")
parser.add_argument(
    '--framework', default='b', choices=['p', 'u', 'b'], help="Choose 'p' for pytest, 'u' for unittest, or 'b' for both (Defaults to both): "
)
parser.add_argument('--model', default='all', help="Path to model to use (Defaults to the most recent epoch for every configuration of hyperparameters): ")
args = parser.parse_args()

framework: TestFrameworkType | Literal['both']
match (getattr(args, 'framework', None)):
    case 'p': framework = 'pytest'
    case 'u': framework = 'unittest'
    case _: framework = 'both'
model_input_dir = getattr(args, "model_input_dir", None)
model_output_dir = getattr(args, "model_output_dir", None)
model_path: Literal['all'] | str = getattr(args, "model", None)

print(f"Running script with:")
print(f"  framework           = {framework}")
print(f"  model_input_dir     = {model_input_dir}")
print(f"  model_output_dir    = {model_output_dir}")
print(f"  model_path          = {model_path}")

# FUNCTIONS

def get_config(config_file):
    """
    Function responsible for getting config information
    :param config_file:
    :return:
    """

    LEARNING_RATE = float(config_file["LEARNING_RATE"])
    EPOCHS = config_file["EPOCHS"]
    BATCH_SIZE = config_file["BATCH_SIZE"]
    TEMPERATURE = float(config_file["TEMPERATURE"])
    EARLY_EPOCH_STOP = config_file["EARLY_EPOCH_STOP"]
    EPOCHS_PER_SAVE = config_file["EPOCHS_PER_SAVE"]
    EMBED_DIM = config_file["EMBED_DIM"]
    HIDDEN_DIM = config_file["HIDDEN_DIM"]
    N_HEADS = config_file["N_HEADS"]
    NUM_LAYERS = config_file["NUM_LAYERS"]
    DROPOUT = float(config_file["DROPOUT"])

    return (
        LEARNING_RATE,
        EPOCHS,
        BATCH_SIZE,
        TEMPERATURE,
        EARLY_EPOCH_STOP,
        EPOCHS_PER_SAVE,
        EMBED_DIM,
        HIDDEN_DIM,
        N_HEADS,
        NUM_LAYERS,
        DROPOUT
    )

def prompt_model(model, tokenizer, test_framework: TestFrameworkType, input_file, output_file):
    """
            Prompts a single model using the input code file

    :param model:
    :param tokenizer:
    :param test_framework:
    :param input_file:
    :param output_file:
    :return:
    """
    if input_file is None:
        print("Must specify input through input file. Cannot input through CLI")
        return

    with open(input_file, "r") as infile:
        prompt = infile.read().strip()

    generated_test = model.generate(
        tokenizer,
        prompt,
        test_framework,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_seq_length=MAX_GEN_SEQ_LEN,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        device=DEVICE
    )

    if output_file is None:
        print("Must specify an output file.")
        return

    with open(output_file, "w") as outfile:
        outfile.write(f"# Generated Unit Test w/ {model.name} for {input_file} using {test_framework}:\n\n")
        outfile.write(generated_test + "\n")

    return generated_test

def prompt_many_models(paths):
    """
    Prompts all the models specified in the arguments
    :param paths:
    :return:
    """
    # Load tokenizer (Same for all)
    tokenizer = Tokenizer(TOKENIZER_PREFIX)
    tokenizer.load()

    print("Loaded tokenizer")
    for p in paths:
        # Need config?
        # (
        #     LEARNING_RATE,
        #     EPOCHS,
        #     BATCH_SIZE,
        #     TEMPERATURE,
        #     EARLY_EPOCH_STOP,
        #     EPOCHS_PER_SAVE,
        #     EMBED_DIM,
        #     HIDDEN_DIM,
        #     N_HEADS,
        #     NUM_LAYERS,
        #     DROPOUT
        # ) = get_config(config_file)

        transformer_model = TransformerEDLanguageModel(
            vocab_size=VOCAB_SIZE
            # vocab_size=VOCAB_SIZE,
            # embed_dim=EMBED_DIM,
            # enc_num_layers=NUM_LAYERS,
            # dec_num_layers=NUM_LAYERS,
            # n_heads=N_HEADS,
            # dropout=DROPOUT,
            # pad_token_id=PAD_TOKEN_ID,
            # seq_len=MAX_TRAIN_SEQ_LEN,
            # name="Transformer Encoder-Decoder"
        ).to(device)

        transformer_model.load_state_dict(torch.load(p))
        print("Didn't need shit for transformer load. Loaded state dict")

        # Collect all relative file paths from MODEL_INPUT_DIR
        for inp in os.listdir(MODEL_INPUT_DIR):
            if os.path.isfile(inp):
                input_file = os.path.join(inp, MODEL_INPUT_DIR)

                # Prompt for each input
                now = time.time()
                fws = []
                for fw in fws:
                    out_file = f"{MODEL_OUTPUT_DIR}/{fw}_for_{input_file}_at_{now.hex()}"
                    print(f"Would do prompt_model({transformer_model}, {tokenizer}, {framework}, {input_file}, {out_file})")
                    # prompt_model(model, tokenizer, framework, input_file, out_file)

# MAIN CODE

# Collect paths for models from args

# For each subdir in TrainingSaves/ we're going to look at the most recent epoch (alphabetically first, by luck) and append to paths
model_paths = []
if model_path == 'all':
    for subdir in os.listdir(TRAINING_SAVE_DIR):
        subdir_path = os.path.join(TRAINING_SAVE_DIR, subdir)
        if os.path.isdir(subdir_path):
            checkpoints = sorted(os.listdir(subdir_path))
            if checkpoints:
                latest_ckpt = checkpoints[0]  # alphabetically first because capitalized
                full_ckpt_path = os.path.join(subdir_path, latest_ckpt)
                rel_path = os.path.relpath(full_ckpt_path, TRAINING_SAVE_DIR)
                model_paths.append(rel_path)
    model_paths = sorted(model_paths)

else:
    model_paths.append(model_path)

print(model_paths)

# Prompt models and save output

prompt_many_models(model_paths)

