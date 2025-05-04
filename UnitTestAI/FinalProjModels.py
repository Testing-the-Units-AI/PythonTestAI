import torch
import torch.nn as nn

class TransformerEDLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, enc_num_layers=6, dec_num_layers=6, dropout=0.2, pad_token_id=0, n_heads=8, seq_len=512, name="TED Model"):
        """ 
        Defines a Encoder-Decoder Transformer model for test use on the Gutenburg project dataset.
        
        :param vocab_size: Vocab size of the tokenizer used. For this project, it was 10,000.
        :param embed_dim: Embedding dimension used to map tokens to embed_dim length vectors.
        :param enc_num_layers: The number of layers for the encoder.
        :param dec_num_layers: The number of layers for the decoder.
        :param dropout: Layer dropout rate used to reduce reliance of specific pathways in node structures by zeroing out nodes.
        :param pad_token_id: Padding ID used by the tokenizer. Ensures the embedding layer has a 0 vector for that index.
        :param n_heads: The number of attention heads used for the Transformer block
        :param seq_len: The maximum sequence length a training sequence can be. Needs to be set for the positional layer to know how many vectors to pre-compute
        :param name: The name which the prompts and graph refer to this model as.
        """
        super().__init__()

        self.name = name

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.positional_embed = PositionalEncoding(embed_dim, dropout=dropout, max_len=seq_len)

        self.transformer = nn.Transformer(d_model=embed_dim,
                                          nhead=n_heads,
                                          num_encoder_layers=enc_num_layers,
                                          num_decoder_layers=dec_num_layers,
                                          dropout=dropout,
                                          batch_first=True)
        
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, enc_input_ids, dec_input_ids, src_padding_mask=None, tgt_padding_mask=None):
        """ 
        The TransformerED implementation of layer propagation. Takes the input to the system with the target data and runs it through a traditional
        encoder-decoder network with embeddings.
        """

        src_embeds = self.embedding(enc_input_ids)
        tgt_embeds = self.embedding(dec_input_ids)

        src_embeds = self.positional_embed(src_embeds)
        tgt_embeds = self.positional_embed(tgt_embeds)

        # Creates an upper triangular matrix full of negative infinity. This stops the model from looking ahead in the sequence and
        # "cheating," seeing the correct token and predicting that token every time. This leads to immediate overfitting and causes
        # prompt completion to repeat the last token it saw for every iteration.
        tgt_seq_len = dec_input_ids.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=enc_input_ids.device)

        out = self.transformer(
            src=src_embeds,
            tgt=tgt_embeds,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        logits = self.fc(out)

        return logits
    
    def generate(self, tokenizer, prompt, max_seq_length=200, bos_token_id=None, eos_token_id=None, pad_token_id=None, temperature=1.0, top_k=4, device='cpu'):
        """ 
        Generates a response, token by token, to the prompt given to the model.

        :param tokenizer: The tokenizer that was used to structure the data for training.
        :param prompt: The input string used to generate a response.
        :param max_seq_length: The max length the response can be, in tokens. Prevents rambling responses from generating infinitely.
        :param eos_token_id: If the eos token is generated, generation ends.
        :param temperature: Determines how creative responses are. Should be high for unit tests generation
        :param device: Device the model should use for generation. Use "cuda" if you have it.
        """

        self.eval()

        # Convert the prompt to tokenized sequence for model comprehension
        enc_input_ids = tokenizer.encode(prompt)
        enc_input_ids = torch.tensor([enc_input_ids], dtype=torch.long, device=device)
        enc_pad_mask = (enc_input_ids == pad_token_id)  # shape (1, src_len)

        # Set up the generation with the bos token id so the model is more aware we are attempting a unit test now
        generated_ids = [bos_token_id] if bos_token_id is not None else []

        # Responses can be a maximum of max_seq_length tokens long
        for _ in range(max_seq_length):
            dec_input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            # Padding mask just to be thorough, doubt its needed since this is generation but cant hurt
            dec_pad_mask = (dec_input_ids == pad_token_id)

            logits = self.forward(
                enc_input_ids,
                dec_input_ids,
                src_padding_mask=enc_pad_mask,
                tgt_padding_mask=dec_pad_mask
            )

            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_token = indices[0, torch.multinomial(probs, num_samples=1)].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            # Check if we are using eos token ID, then ask if what was generated was that ID so we can end early
            if eos_token_id is not None and next_token == eos_token_id:
                break
            # Build our response with next token
            generated_ids.append(next_token)

        # Once we have an eos token or the max length is reached, return the decoded prompt
        gen_ids = generated_ids[1:] if bos_token_id is not None else generated_ids
        return tokenizer.decode(gen_ids)



#Extra layer provided by Dr. James Ghawaly for the positional embedding needed for transformer models to understand token order
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Upper triangular negative infinity mask generation
def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
