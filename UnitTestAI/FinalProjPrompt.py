import argparse
import json
import time

import torch
import os
from typing import Literal, List

from FinalProjConstants import MODEL_INPUT_DIR, MODEL_OUTPUT_DIR, MAX_GEN_SEQ_LEN, TOP_K, DEVICE, TRAINING_SAVE_DIR, \
    device, TOKENIZER_PREFIX, VOCAB_SIZE, MAX_TRAIN_SEQ_LEN, CONFIG_FILE
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

def get_configs(config_file):
    """
    Function responsible for getting config information from file
    :param config_file:
    :return:
    """
    with open(config_file, 'r') as file:
        content = file.read().strip()
        json_content = json.loads(content)
        print(f"Loaded content from {CONFIG_FILE}")
        return json_content

def parse_path(path):
    """
        Eg "./TrainingSaves/Epochs_3_Batch_Size_128_Temp_0.8_Learning_0.004_Layers_4_Dropout_0.2/TrainLoss_4.7667_TestLoss_5.7543_Perplexity_117.5363_BLEU_0.0001"
        epochs = 3, batch_size = 128, temperature = 0.8, learning_rate = 0.004, layers = 4, dropout = 0.2, and ignores stuff after "/"

    :param path: Path to model
    :return: Tuple of (epochs, batch_size, temperature, learning_rate, layers, dropout)
    """
    # Only care about the part before first slash
    rel = os.path.relpath(path, TRAINING_SAVE_DIR)
    prefix = rel.split("/")[0]

    # Split by underscores
    tokens = prefix.split("_")

    # Dictionary to hold parsed values
    params = {}

    # Walk through tokens pairwise
    i = 0
    while i < len(tokens) - 1:
        key = tokens[i].lower()
        value = tokens[i+1]

        if key == "epochs":
            params["epochs"] = int(value)
        elif key == "batch":
            params["batch_size"] = int(value)
        elif key == "temp":
            params["temperature"] = float(value)
        elif key == "learning":
            params["learning_rate"] = float(value)
        elif key == "layers":
            params["layers"] = int(value)
        elif key == "dropout":
            params["dropout"] = float(value)

        i += 2  # Move to next pair

    return (
        params["epochs"],
        params["batch_size"],
        params["temperature"],
        params["learning_rate"],
        params["layers"],
        params["dropout"]
    )

def dissect_config(path, configs: List[dict[str, str]]) -> dict[str, str] | None:
    """

    :param path: Path to model
    :param configs: Configs obtained from config file before prompting models
    :return:
    """
    # parse path for info
    (
        epochs,
        batch_size,
        temperature,
        learning_rate,
        layers,
        dropout
    ) = parse_path(path)

    # find right config from configs
    config = None
    for c in configs:
        # print("Checking against config:\n", json.dumps(c))
        if int(c["EPOCHS"]) == epochs and int(c["BATCH_SIZE"]) == batch_size and float(c["TEMPERATURE"]) == temperature and float(c["LEARNING_RATE"]) == learning_rate and int(c["NUM_LAYERS"]) == layers and float(c["DROPOUT"]) == dropout:
            config = c
            break

    return config

def prompt_model(model, model_config, tokenizer, test_framework: TestFrameworkType, input_file, output_file):
    """
            Prompts a single model using the input code file

    :param model:
    :param model_config:
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
        temperature=model_config['temperature'],
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

def prompt_many_models(paths, configs):
    """
    Prompts all the models specified in the arguments
    :param paths:
    :param configs:
    :return:
    """
    # Load tokenizer (Same for all)
    tokenizer = Tokenizer(TOKENIZER_PREFIX)
    tokenizer.load()
    print("Loaded tokenizer")

    for p in paths:
        # Config and construct model so we can use saved weights & biases
        # p has information about model config: dissect it so we can get right config using what we know
        model_config = dissect_config(p, configs)

        transformer_model = TransformerEDLanguageModel(
            vocab_size=VOCAB_SIZE,
            enc_num_layers=model_config["NUM_LAYERS"],
            dec_num_layers=model_config["NUM_LAYERS"],
            pad_token_id=PAD_TOKEN_ID,
            seq_len=MAX_TRAIN_SEQ_LEN,
            name=f"Transformer {os.path.dirname(p)}",

        ).to(device)

        transformer_model.load_state_dict(torch.load(p))
        print(f"Loaded state dict: {transformer_model.name}")

        # Collect all relative file paths from MODEL_INPUT_DIR
        for inp in os.listdir(MODEL_INPUT_DIR):
            if os.path.isfile(inp):
                input_file = os.path.join(inp, MODEL_INPUT_DIR)

                # Prompt for each input
                now = time.time()
                fws = []
                for fw in fws:
                    out_file = f"{MODEL_OUTPUT_DIR}/{fw}_for_{input_file}_at_{now.hex()}"
                    print(f"Would do prompt_model(\n{transformer_model}, \n{model_config}, \n{tokenizer}, \n{framework}, \n{input_file}, \n{out_file})")
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
                latest_ckpt = checkpoints[0]  # alphabetically first
                full_ckpt_path = os.path.join(subdir_path, latest_ckpt)
                rel_path = os.path.relpath(full_ckpt_path, "./")  # needs to be relative './' otherwise screwed
                model_paths.append(rel_path)
    model_paths = sorted(model_paths)

else:
    model_paths.append(model_path)

print(model_paths)

prompt_many_models(model_paths, get_configs(CONFIG_FILE))
