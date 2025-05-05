import argparse
import os

import torch

from FinalProject import MODEL_INPUT_DIR, MODEL_OUTPUT_DIR, transformer_model, prompt_model, tokenizer
from FinalProjModels import TestFrameworkType

parser = argparse.ArgumentParser()
parser.add_argument(
    '--framework', default='u', choices=['p', 'u'], help="Choose 'p' for pytest or 'u' for unittest (default: 'u')"
)
parser.add_argument('--input_code_file', default=None, help="Where do you want to get code to generate tests for?")
parser.add_argument('--output_test_file', default=None, help="Where do you want to output tests?")
args = parser.parse_args()

train_new_tokenizer = getattr(args, "train_new_tokenizer", False)
train_new_model = getattr(args, "train_new_model", False)
old_model = getattr(args, "old_model_dir", "")
framework: TestFrameworkType = "unittest" if getattr(args, "framework", None) == 'u' else "pytest"
input_code_file = getattr(args, "input_code_file", None)
output_test_file = getattr(args, "output_test_file", None)

print(f"Running script with:")
print(f"  train_new_tokenizer = {train_new_tokenizer}")
print(f"  train_new_model     = {train_new_model}")
print(f"  framework           = {framework}")
print(f"  input_code_file     = {input_code_file}")
print(f"  output_test_file    = {output_test_file}")


old_model_paths = [
    "Epochs_3_Batch_Size_128_Temp_0.8_Learning_0.004_Layers_4_Dropout_0.2/checkpoint_epoch_3.pth",
    "Epochs_3_Batch_Size_128_Temp_0.9_Learning_0.002_Layers_4_Dropout_0.2/checkpoint_epoch_3.pth",
    "Epochs_3_Batch_Size_128_Temp_1.1_Learning_0.001_Layers_4_Dropout_0.2/checkpoint_epoch_3.pth",
    "Epochs_6_Batch_Size_128_Temp_0.8_Learning_0.002_Layers_4_Dropout_0.3/checkpoint_epoch_4.pth",
    "Epochs_6_Batch_Size_128_Temp_1.1_Learning_0.001_Layers_4_Dropout_0.2/checkpoint_epoch_3.pth",
    "Epochs_12_Batch_Size_128_Temp_0.9_Learning_0.002_Layers_4_Dropout_0.2/checkpoint_epoch_5.pth"
]

# TRANSFORMER ED

def gen_unit_tests_per_model(path):
    for config in configs:
        BATCH_SIZE = config["BATCH_SIZE"]
        EPOCHS = config["EPOCHS"]
        LEARNING_RATE = float(config["LEARNING_RATE"])
        TEMPERATURE = float(config["TEMPERATURE"])
        EARLY_EPOCH_STOP = config["EARLY_EPOCH_STOP"]
        EPOCHS_PER_SAVE = config["EPOCHS_PER_SAVE"]
        EMBED_DIM = config["EMBED_DIM"]
        HIDDEN_DIM = config["HIDDEN_DIM"]
        NUM_LAYERS = config["NUM_LAYERS"]
        DROPOUT = float(config["DROPOUT"])
        N_HEADS = config["N_HEADS"]

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

    transformer_model.load_state_dict(torch.load(path))
