import torch

from PythonTestAI.trainer import train_model, load_tokenizer
from models import TransformerLanguageModel
from constants import *

device = torch.device(DEVICE)

tokenizer = load_tokenizer(TOKENIZER_PATH)

transformer_model = TransformerLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    n_heads=N_HEADS,
    dropout=DROPOUT,
    pad_token_id=PAD_TOKEN_ID,
    top_p=TOP_P,
    name="Transformer"
).to(device)

train_model(transformer_model, device, tokenizer, transformer_model.name)