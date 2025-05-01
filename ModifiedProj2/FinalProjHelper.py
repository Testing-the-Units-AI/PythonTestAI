import os
import json
import torch
from torch.utils.data import Dataset

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

        # This is just the input to the encoder to give extra context to our decoder
        encoder_input_ids = torch.tensor(src_tokens, dtype=torch.long)
        # Creating the inputs and labels such that we can train by teacher forcing. Note, the decode index
        # is the prefix (what we are given) and the label is shifted one forward, giving what we want to predict.
        decoder_input_ids = torch.tensor(tgt_tokens[:-1], dtype=torch.long) #******self.tokenizer.bos_token_id +
        labels = torch.tensor(tgt_tokens[1:], dtype=torch.long) #******+ [tokenizer.eos_token_id]
        return encoder_input_ids, decoder_input_ids, labels
