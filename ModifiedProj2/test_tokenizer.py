import pytest
from FinalProjHelper import Tokenizer


jsonl = {
    "code": "def is_koish(board, c):\n    \"\"\"Check if c is surrounded on all sides by 1 color, and return that color\"\"\"\n    if board[c] != EMPTY:\n        return None\n    neighbors = {board[n] for n in NEIGHBORS[c]}\n    if len(neighbors) == 1 and (not EMPTY in neighbors):\n        return list(neighbors)[0]\n    else:\n        return None\ndef set_board_size(n): ...\ndef place_stones(board, color, stones): ...\ndef find_reached(board, c): ...\ndef is_koish(board, c): ...\ndef is_eyeish(board, c): ...\n",
    "test": "def test_is_koish(self):\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('A9')), BLACK)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('B8')), None)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('B9')), None)\n    self.assertEqual(go.is_koish(TEST_BOARD, pc('E5')), None)",
    "framework": "unittest"
}

@pytest.fixture
def tokenizer():
    return Tokenizer()

def test_encode_returns_pieces(tokenizer):
    # Requires bpe_model.model to exist
    code = "def hello():\n    print('hi')"
    pieces = tokenizer.encode(code)
    assert isinstance(pieces, list)
    assert all(isinstance(p, str) for p in pieces)
