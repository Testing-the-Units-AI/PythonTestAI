import argparse

import torch


from FinalProject import prompt_model, DEVICE, VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, N_HEADS, DROPOUT, \
    MAX_TRAIN_SEQ_LEN
from FinalProjHelper import Tokenizer, PAD_TOKEN_ID
from FinalProjModels import TestFrameworkType

# Test prompt model: load in model, read input from file, store output in file (as expected by arg parse)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--framework',
    default='u',
    choices=['p', 'u'],
    help="Choose 'p' for pytest or 'u' for unittest (default: 'u')"
)
parser.add_argument('--input_code_file', default=None, help="Where do you want to get code to generate tests for?")
parser.add_argument('--output_test_file', default=None, help="Where do you want to output tests?")

args = parser.parse_args()

framework: TestFrameworkType = 'unittest' if args.framework == 'u' else 'pytest'
input_code_file = args.input_code_file
output_test_file = args.output_test_file

print(f"Running prompt_test  with:")
print(f"  framework           = {framework}")
print(f"  input_code_file     = {input_code_file}")
print(f"  output_test_file    = {output_test_file}")


# Tokenizer (don't train again)
tokenizer = Tokenizer('bpe_model')
device = torch.device(DEVICE)
tokenizer.load()

# Model (don't train again)

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

transformer_model.load_state_dict(torch.load('./TrainingSaves/final_model_weights.pth'))

prompt_model(transformer_model, tokenizer, framework, input_code_file, output_test_file)

# TODO: These should be reasonable if prompt_model works
with open(input_code_file, 'r') as in_artifact:
    inp = in_artifact.read().strip()

    print(f'inp: {inp}')

    assert len(inp) > 5

with open(output_test_file, 'r') as out_artifact:
    out = out_artifact.read().strip()

    print(f'out: {out}')

    assert len(out) > 5
