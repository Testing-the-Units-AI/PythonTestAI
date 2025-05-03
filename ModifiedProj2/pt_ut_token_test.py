from FinalProjHelper import TextDatasetTED
from FinalProjHelper import Tokenizer
from FinalProjHelper import *


tokenizer = Tokenizer('test')

# tokenizer.train(jsonl_file='data/dataset.jsonl')

tokenizer.load()

for v in range(min(20,tokenizer.get_piece_size())):
    print(tokenizer.sp.IdToPiece(v))

assert tokenizer.sp.IdToPiece(BOS_TOKEN_ID) == BOS_TOKEN
assert tokenizer.sp.IdToPiece(EOS_TOKEN_ID) == EOS_TOKEN
assert tokenizer.sp.IdToPiece(PAD_TOKEN_ID) == PAD_TOKEN

assert tokenizer.sp.IdToPiece(PYTEST_TOKEN_ID) == PYTEST_TOKEN
assert tokenizer.sp.IdToPiece(UNITTEST_TOKEN_ID) == UNITTEST_TOKEN

print("All tokenizer tests passed")

dataset = TextDatasetTED('data/dataset.jsonl',tokenizer)

encountered_pytest = False
encountered_unittest = False
for i, batch in enumerate(dataset):
    if i > 10:
        break
    enc, dec, lab = batch

    bos_id = dec[0].item()
    fw_tok_id = dec[1].item()
    eos_id = lab[-1].item()
    lab_fw_tid = lab[0].item()
    assert bos_id == BOS_TOKEN_ID
    assert fw_tok_id == PYTEST_TOKEN_ID or fw_tok_id == UNITTEST_TOKEN_ID
    assert eos_id == EOS_TOKEN_ID
    assert lab_fw_tid == PYTEST_TOKEN_ID or fw_tok_id == UNITTEST_TOKEN_ID

    if fw_tok_id == PYTEST_TOKEN_ID:
        encountered_pytest = True
    elif fw_tok_id == UNITTEST_TOKEN_ID:
        encountered_unittest = True

print("Encountered both pytest and ut: ", encountered_pytest and encountered_unittest)