import json
from io import BytesIO
import tokenize

jsonl = {
    "code": "def is_koish(board, c):\n    \"\"\"Check if c is surrounded on all sides by 1 color, and return that color\"\"\"\n    if board[c] != EMPTY:\n        return None\n    neighbors = {board[n] for n in NEIGHBORS[c]}\n    if len(neighbors) == 1 and (not EMPTY in neighbors):\n        return list(neighbors)[0]\n    else:\n        return None\ndef set_board_size(n): ...\ndef place_stones(board, color, stones): ...\ndef find_reached(board, c): ...\ndef is_koish(board, c): ...\ndef is_eyeish(board, c): ...\n",
    "test": "def test_is_koish(self):\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('A9')), BLACK)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('B8')), None)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('B9')), None)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('E5')), None)",
    "framework": "unittest"
}

from io import BytesIO
import tokenize

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

code_tokens = tokenize_string_with_structure(jsonl["code"])
test_tokens = tokenize_string_with_structure(jsonl["test"])
to_bpe = " ".join(code_tokens + test_tokens)

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

# Load trained model
sp = spm.SentencePieceProcessor()
sp.Load("bpe_model.model")

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

    return "".join(out).lstrip()

# Example usage
# Use structured token stream, not raw code
code_tokens_structured = tokenize_string_with_structure(jsonl["code"])
code_text_for_encoding = " ".join(code_tokens_structured)
code_pieces = sp.Encode(code_text_for_encoding, out_type=str)

# Reconstruct readable code
print(reconstruct_code(code_pieces, num_indents))

ids = sp.EncodeAsIds(code_text_for_encoding)
print(ids)


#  Try running on more of set

# Accumulate all structured tokens here
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

        code_tokens = tokenize_string_with_structure(code)
        test_tokens = tokenize_string_with_structure(test)

        all_tokens.extend(code_tokens + test_tokens)

# Join into one string for SentencePiece training
to_bpe = " ".join(all_tokens)

# Save for SentencePiece
train_tokenizer("<BOS>" + to_bpe + "<EOS>")

# for i in range(sp.vocab_size()):
#     print(f"Token {i}: {sp.Decode(i)}")

sp.Load("bpe_model.model")

# Use structured token stream, not raw code
code_tokens_structured = tokenize_string_with_structure(jsonl["code"])
# code_text_for_encoding = " ".join(code_tokens_structured)
code_pieces = sp.Encode(code_text_for_encoding, out_type=str)

token_ids = sp.EncodeAsIds(code_tokens_structured)
flat = [item for sublist in token_ids for item in sublist]
print('ids: ', flat)

reconstructed = reconstruct_code(code_pieces, num_indent_spaces=4)
print(reconstructed)