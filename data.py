import torch
import random
from torchtext import data
from torchtext import datasets


def getdata():
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    MAX_VOCAB_SIZE = 25_000
    BATCH_SIZE = 64
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    prem = TEXT.vocab.vectors
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)
    INPUT_DIM = len(TEXT.vocab)
    return train_iterator, valid_iterator, test_iterator, INPUT_DIM,prem,pad_idx,unk_idx,TEXT,LABEL



