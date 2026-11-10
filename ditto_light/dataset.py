import torch

from torch.utils import data
from transformers import AutoTokenizer

# map lm name to huggingface's pre-trained model names
lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased'
}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta'):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)

        return x, self.labels[idx]


    def pad(self, batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x of shape (batch_size, seq_len)
            LongTensor: a batch of labels, (batch_size,)
        """
        pad_id = self.tokenizer.pad_token_id
        x, y = zip(*batch)
        maxlen = max(len(xi) for xi in x)
        x = [xi + [pad_id] * (maxlen - len(xi)) for xi in x]

        return torch.LongTensor(x), torch.LongTensor(y)

    def add_new_examples(self, examples):
        # add new examples (e.g., from augmentation) to the dataset
        for ex in examples:
                s1, s2, label = ex.strip().split('\t')
                self.pairs.append((s1, s2))
                self.labels.append(int(label))
        return len(self.pairs)
