import os
import json
from typing import List

import torch
from torch.utils.data import Dataset
import tokenize
from io import BytesIO
import sentencepiece as spm
from tqdm import tqdm

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
PYTEST_TOKEN = "<PYTEST>"
UNITTEST_TOKEN = "<UNITTEST>"
BOS_TOKEN_ID = 3
EOS_TOKEN_ID = 4
PAD_TOKEN_ID = 5
PYTEST_TOKEN_ID = 9
UNITTEST_TOKEN_ID = 10

#
# # For tokenizer training
# def merge_text_files(directory, outfile_name):
#     """
#     This function takes all of the raw data and uses it to train the bpe tokenizer such that,
#     when we have our eventual prompts and answers, we can easily convert them into tokenized sequences
#     for training our models and decode them for reading. BPE tokenization allows for the token sequences to be condensed
#     to small sequences of letters or even full words for easier processing and better pattern recognition.
#
#     :param directory: The folder containing all of the raw data.
#     :param outfile_name: The corpus file name to write to
#     """
#
#
#     # # We want a clean file when we call this function
#     # if os.path.exists(outfile_name):
#     #     os.remove(outfile_name)
#     #
#     # # We will merge all of the text files in the specified directory into one file so it can be used for training the tokenizer
#     # with open(outfile_name, 'w', encoding='utf-8') as out:
#     #     for file in os.listdir(directory):
#     #         filename = os.fsdecode(file)
#     #         if filename.endswith('.txt'):
#     #             file_path = os.path.join(directory, filename)
#     #             with open(file_path, 'r', encoding='utf-8') as cur_data:
#     #                 out.write(cur_data.read())
#     #                 out.write('\n')

class Tokenizer:

    def __init__(self, tokenizer_prefix):
        self.tokenizer_prefix = tokenizer_prefix
        self.sp = spm.SentencePieceProcessor()

    def load(self):
        self.sp.Load(f"./TokenizerModels/{self.tokenizer_prefix}.model")

    def _reconstruct_code(self, pieces, num_indent_spaces = 4):
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
    def _tokenize_string_with_structure(self, source_code):
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

    def train(self, vocab_size = 7017, jsonl_file = "./data/all.jsonl", sample_limit: int | None = 10000):
        # Make tokenizer
        all_tokens = []

        with open(jsonl_file, mode="r") as data_file:
            for i, line in enumerate(data_file):
                if sample_limit and i > sample_limit:
                    break
                if not line.strip():
                    continue
                jsonl = json.loads(line)
                code = jsonl.get("code", "")
                test = jsonl.get("test", "")

                code_tokens = self._tokenize_string_with_structure(code)
                test_tokens = self._tokenize_string_with_structure(test)

                all_tokens.extend(code_tokens + [BOS_TOKEN] + test_tokens + [EOS_TOKEN])

        to_bpe = " ".join(all_tokens)

        temp_file = "data/token_input.txt"
        with open(temp_file, "w") as f:
            f.write(to_bpe)
        print("Temp file for tokenizer input created")

        spm.SentencePieceTrainer.Train(
            input=temp_file,
            model_prefix=self.tokenizer_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            user_defined_symbols=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, '\\n', '<INDENT>', '<DEDENT>', PYTEST_TOKEN, UNITTEST_TOKEN],
            num_threads=8
        )

        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("Temp file deleted.")
        else:
            print("Temp file was not found (nothing happened, but this option's unlikely).")

    def encode(self, encode_this: str) -> List[int]:
        # Use structured token stream, not raw code
        code_tokens_structured = " ".join(self._tokenize_string_with_structure(source_code=encode_this))
        return self.sp.EncodeAsIds(code_tokens_structured)

    def encodeAsPieces(self, encode_this: str) -> List[int]:
                # Use structured token stream, not raw code
        code_tokens_structured = " ".join(self._tokenize_string_with_structure(source_code=encode_this))
        print('return (pieces, ids) ')
        return self.sp.EncodeAsPieces(code_tokens_structured)


    def decode(self, decode_this: List[int]) -> str:
        pieces = [self.sp.IdToPiece(id) for id in decode_this]

        reconstructed = self._reconstruct_code(pieces, num_indent_spaces=4)
        return reconstructed

    def get_piece_size(self):
        self.load()
        return self.sp.GetPieceSize()

class TextDatasetTED(Dataset):
    def __init__(self, filepath, tokenizer, max_src_len=128, max_tgt_len=32):
        """


        :param filepath: Location of the .jsonl file with prompts and completions
        :param tokenizer: Tokenizer we want to use for the training
        :param max_src_len: A limit for how long we will allow a prompt to be from the .jsonl
        :param max_tgt_len: A limit for how long we will allow a label to be from the .jsonl
        """

        self.samples = []
        self.test_framework = []
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # With the provided jsonl files from the handout...
        with open(filepath, "r", encoding="utf-8") as file:
            # Line by line (with each line being a prompt/completion pair)...
            for line in tqdm(file, desc="Tokenizing dataset", dynamic_ncols=True, leave=True, ascii=True):
                item = json.loads(line)
                src_tokens = tokenizer.encode(item["code"])[:max_src_len]
                tgt_tokens = tokenizer.encode(item["test"])[:max_tgt_len]
                fw = item["framework"]

                # We also don't want lines that are too short and don't provide useful info
                if len(src_tokens) < 2:
                    continue
                # Now that our jsonl line is tokenized, record the result
                # Note that is is a JAGGED 2D array, meaning we will need to add padding later
                self.samples.append((src_tokens, tgt_tokens))
                self.test_framework.append(fw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Takes the samples from initialization and modifies them for teacher forcing for the models

        :param index: The sample being references
        :return: The inputs and target for that sample
        """

        src_tokens, tgt_tokens = self.samples[index]

        fw_token_id = PYTEST_TOKEN_ID if self.test_framework[index] == "pytest" else UNITTEST_TOKEN_ID

        tgt_tokens = [BOS_TOKEN_ID, fw_token_id] + tgt_tokens + [EOS_TOKEN_ID]

        # This is just the input to the encoder to give extra context to our decoder
        encoder_input_ids = torch.tensor(src_tokens, dtype=torch.long)
        # Creating the inputs and labels such that we can train by teacher forcing. Note, the decode index
        # is the prefix (what we are given) and the label is shifted one forward, giving what we want to predict.
        decoder_input_ids = torch.tensor(tgt_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tgt_tokens[1:], dtype=torch.long)
        return encoder_input_ids, decoder_input_ids, labels
