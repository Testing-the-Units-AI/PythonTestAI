import json

from PythonTestAI.ModifiedProj2.FinalProjHelper import Tokenizer

EOS_TOKEN = "<EOS>"
BOS_TOKEN = "<BOS>"

jsonl = {
    "code": "def is_koish(board, c):\n    \"\"\"Check if c is surrounded on all sides by 1 color, and return that color\"\"\"\n    if board[c] != EMPTY:\n        return None\n    neighbors = {board[n] for n in NEIGHBORS[c]}\n    if len(neighbors) == 1 and (not EMPTY in neighbors):\n        return list(neighbors)[0]\n    else:\n        return None\ndef set_board_size(n): ...\ndef place_stones(board, color, stones): ...\ndef find_reached(board, c): ...\ndef is_koish(board, c): ...\ndef is_eyeish(board, c): ...\n",
    "test": "def test_is_koish(self):\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('A9')), BLACK)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('B8')), None)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('B9')), None)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('E5')), None)",
    "framework": "unittest"
}

from io import BytesIO
import tokenize


# Useful because tokenizer needs to preserve structure of code!
def tokenize_string_with_structure(source_code):
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

import sentencepiece as spm

# Actually train it
def train_tokenizer(to_bpe):

    with open("token_input.txt", "w") as f:
        f.write(to_bpe)

    spm.SentencePieceTrainer.Train(
        input="token_input.txt",
        model_prefix="bpe_model",
        vocab_size=7017,               # Set your desired vocab size
        model_type="bpe",              # You can also try 'unigram', 'char', etc.
        character_coverage=1.0,         # 1.0 means full coverage
        user_defined_symbols=['\\n', '<INDENT>', '<DEDENT>', '<BOS>', '<EOS>'],
        num_threads = 8  # 👈 Use your desired number of threads
    )



# For inference! Take token outputs and put it into English (Python)
num_indents = 4
def reconstruct_code(pieces, num_indent_spaces):
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



### Example usage


# Load trained model
sp = spm.SentencePieceProcessor()
sp.Load("./bpe_model.model")

# Sanity check
print("Vocab contains:")
has_newline = False
has_indent = False
has_dedent = False

for i in range(sp.vocab_size()):
    piece = sp.id_to_piece(i)
    if piece == '\\n':
        has_newline = True
    elif piece == '<INDENT>':
        has_indent = True
    elif piece == '<DEDENT>':
        has_dedent = True

print("  \\n      :", has_newline)
print("  <INDENT>:", has_indent)
print("  <DEDENT>:", has_dedent)



tokenizer = Tokenizer()
# Use structured token stream, not raw code
code_tokens_structured = tokenize_string_with_structure(jsonl["code"])
pleasantly_tokenized_with_struct = tokenizer.tokenize_string_with_structure(jsonl["code"])

assert pleasantly_tokenized_with_struct == code_tokens_structured

hard_tokenized_code = sp.EncodeAsPieces(" ".join(code_tokens_structured))

pleasantly_tokenized = tokenizer.encodeAsPieces(jsonl["code"])
assert pleasantly_tokenized == hard_tokenized_code
print("Tokenized Assert passed")

# Reconstruct readable code
hard_reconstructed = reconstruct_code(hard_tokenized_code, num_indents)
pleasantly_reconstructed = tokenizer.reconstruct_code(pleasantly_tokenized)

print('Hard reconstr: ', hard_reconstructed)

assert pleasantly_reconstructed == hard_reconstructed
print("Reconstruction Assert passed")

def flatten(nested):
    return [item for sublist in nested for item in sublist]


ids = sp.EncodeAsIds(code_tokens_structured)
ids = flatten(ids)

pleasantly_ids = tokenizer.encode(jsonl["code"])

assert ids == pleasantly_ids
print("Encode ID assert passed")


print(tokenizer.decode(pleasantly_ids))

# # Make tokenizer
# all_tokens = []
#
# max_samples = 5000
# with open("./data/all.jsonl", mode="r") as data_file:
#     for i, line in enumerate(data_file):
#         if i > max_samples:
#             break
#         if not line.strip():
#             continue
#         jsonl = json.loads(line)
#         code = jsonl.get("code", "")
#         test = jsonl.get("test", "")
#
#         code_tokens = tokenize_string_with_structure(code)
#         test_tokens = tokenize_string_with_structure(test)
#
#         all_tokens.extend(code_tokens + [BOS_TOKEN] + test_tokens + [EOS_TOKEN])
#
# to_bpe = " ".join(all_tokens)
# train_tokenizer(to_bpe)


sp.Load("bpe_model.model")

# Use structured token stream, not raw code
code_tokens_structured = tokenize_string_with_structure(jsonl["code"])

# Encode
code_pieces = flatten(sp.EncodeAsPieces(code_tokens_structured))
token_ids = flatten(sp.EncodeAsIds(code_tokens_structured))

print('ids:', token_ids)

# Reconstruct from pieces
reconstructed_from_pieces = reconstruct_code(code_pieces, num_indent_spaces=4)
print("From pieces:\n", reconstructed_from_pieces)

# ✅ Convert IDs back to pieces
decoded_pieces = [sp.IdToPiece(i) for i in token_ids]

# Reconstruct from decoded pieces
reconstructed_from_ids = reconstruct_code(decoded_pieces, num_indent_spaces=4)
print("From ids:\n", reconstructed_from_ids)


