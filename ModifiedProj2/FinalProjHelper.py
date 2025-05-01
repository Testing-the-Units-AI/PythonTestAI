import os
import json
from typing import List

import torch
from torch.utils.data import Dataset
import tokenize
from io import BytesIO
import sentencepiece as spm

from PythonTestAI.ModifiedProj2.FinalProject import BOS_TOKEN, BOS_TOKEN_ID, EOS_TOKEN_ID


# For tokenizer training
def merge_text_files(directory, outfile_name):
    """
    This function takes all of the raw data and uses it to train the bpe tokenizer such that,
    when we have our eventual prompts and answers, we can easily convert them into tokenized sequences
    for training our models and decode them for reading. BPE tokenization allows for the token sequences to be condensed
    to small sequences of letters or even full words for easier processing and better pattern recognition.

    :param directory: The folder containing all of the raw data.
    :param outfile_name: The corpus file name to write to
    """
    # We want a clean file when we call this function
    if os.path.exists(outfile_name):
        os.remove(outfile_name)

    # We will merge all of the text files in the specified directory into one file so it can be used for training the tokenizer
    with open(outfile_name, 'w', encoding='utf-8') as out:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as cur_data:
                    out.write(cur_data.read())
                    out.write('\n')

class Tokenizer:

    def __init__(self):
        pass

    def reconstruct_code(self, pieces, num_indent_spaces = 4):
        indent_level = 0
        indentation = " " * num_indent_spaces

        out = []
        current_line = []

        for p in pieces:
            if p == '\\n':
                # Flush current line with proper indentation
                if current_line:
                    out.append(indentation * indent_level + "".join(current_line).lstrip() + "\n")
                    current_line = []
                else:
                    out.append("\n")
            elif p == '<INDENT>':
                indent_level += 1
            elif p == '<DEDENT>':
                indent_level = max(0, indent_level - 1)
            else:
                if p.startswith('▁'):
                    current_line.append(' ' + p[1:])
                else:
                    current_line.append(p)

        # Catch any remaining line
        if current_line:
            out.append(indentation * indent_level + "".join(current_line).lstrip())

        return " ".join(out).lstrip()

    # Useful because tokenizer needs to preserve structure of code!
    def tokenize_string_with_structure(self, source_code):
        tokens = tokenize.tokenize(BytesIO(source_code.encode('utf-8')).readline)
        out = []

        for tok in tokens:
            if tok.type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            elif tok.type == tokenize.NEWLINE:
                out.append('\\n')
            elif tok.type == tokenize.INDENT:
                out.append('<INDENT>')
            elif tok.type == tokenize.DEDENT:
                out.append('<DEDENT>')
            elif tok.type == tokenize.NL:
                out.append('\\n')  # For blank lines or non-logical newlines
            else:
                out.append(tok.string)

        return out

    def train(self, bos_token, eos_token):

        # Make tokenizer
        all_tokens = []

        max_samples = 10000
        with open("./data/all.jsonl", mode="r") as data_file:
            for i, line in enumerate(data_file):
                if i > max_samples:
                    break
                if not line.strip():
                    continue
                jsonl = json.loads(line)
                code = jsonl.get("code", "")
                test = jsonl.get("test", "")

                code_tokens = self.tokenize_string_with_structure(code)
                test_tokens = self.tokenize_string_with_structure(test)

                all_tokens.extend(code_tokens + [bos_token] + test_tokens + [eos_token])

        to_bpe = " ".join(all_tokens)
        with open("token_input.txt", "w") as f:
            f.write(to_bpe)

        spm.SentencePieceTrainer.Train(
            input="token_input.txt",
            model_prefix="bpe_model",
            vocab_size=7017,  # Set your desired vocab size
            model_type="bpe",  # You can also try 'unigram', 'char', etc.
            character_coverage=1.0,  # 1.0 means full coverage
            user_defined_symbols=['\\n', '<INDENT>', '<DEDENT>', '<BOS>', '<EOS>'],
            num_threads=8  # 👈 Use your desired number of threads
        )

    def encode(self, encode_this: str) -> List[int]:
        sp = spm.SentencePieceProcessor()
        sp.Load("bpe_model.model")

        # Use structured token stream, not raw code
        code_tokens_structured = " ".join(self.tokenize_string_with_structure(source_code=encode_this))
        return sp.EncodeAsIds(code_tokens_structured)

    def encodeAsPieces(self, encode_this: str) -> List[int]:
        sp = spm.SentencePieceProcessor()
        sp.Load("bpe_model.model")

        # Use structured token stream, not raw code
        code_tokens_structured = " ".join(self.tokenize_string_with_structure(source_code=encode_this))
        print('return (pieces, ids) ')
        return sp.EncodeAsPieces(code_tokens_structured)


    def decode(self, decode_this: List[int]) -> str:
        sp = spm.SentencePieceProcessor()
        sp.Load("bpe_model.model")

        pieces = [sp.IdToPiece(id) for id in decode_this]

        reconstructed = self.reconstruct_code(pieces, num_indent_spaces=4)
        return reconstructed

    def get_piece_size(self):
        sp = spm.SentencePieceProcessor()
        sp.Load("bpe_model.model")
        return sp.GetPieceSize()

class TextDatasetTED(Dataset):
    def __init__(self, filepath, tokenizer, max_src_len=128, max_tgt_len=32):
        """ 
        

        :param filepath: Location of the .jsonl file with prompts and completions
        :param tokenizer: Tokenizer we want to use for the training
        :param max_src_len: A limit for how long we will allow a prompt to be from the .jsonl
        :param max_tgt_len: A limit for how long we will allow a label to be from the .jsonl
        """

        self.samples = []
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # With the provided jsonl files from the handout...
        with open(filepath, "r", encoding="utf-8") as file:
            # Line by line (with each line being a prompt/completion pair)...
            for line in file:
                item = json.loads(line)
                src_tokens = tokenizer.encode(item["prompt"], out_type=int)[:max_src_len]
                tgt_tokens = tokenizer.encode(item["completion"], out_type=int)[:max_tgt_len]

                # We also don't want lines that are too short and don't provide useful info
                if len(src_tokens) < 2:
                    continue
                # Now that our jsonl line is tokenized, record the result
                # Note that is is a JAGGED 2D array, meaning we will need to add padding later
                self.samples.append((src_tokens, tgt_tokens))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """ 
        Takes the samples from initialization and modifies them for teacher forcing for the models

        :param index: The sample being references
        :return: The inputs and target for that sample
        """

        src_tokens, tgt_tokens = self.samples[index]

        tgt_tokens = [BOS_TOKEN_ID] + tgt_tokens + [EOS_TOKEN_ID]

        # This is just the input to the encoder to give extra context to our decoder
        encoder_input_ids = torch.tensor(src_tokens, dtype=torch.long)
        # Creating the inputs and labels such that we can train by teacher forcing. Note, the decode index
        # is the prefix (what we are given) and the label is shifted one forward, giving what we want to predict.
        decoder_input_ids = torch.tensor(tgt_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tgt_tokens[1:], dtype=torch.long)
        return encoder_input_ids, decoder_input_ids, labels
