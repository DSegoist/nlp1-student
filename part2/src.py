from pathlib import Path
from urllib.request import urlretrieve
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from itertools import combinations, product

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
 
import part1
from part1.clslib import Encoder

 
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
TOKENIZER_PATH = RESULTS_DIR / "sst2_tokenizer.json"
 
CLASSES = part1.clslib.NamedClasses(['negative', 'positive'])
 
NUM_FOLDS = 5
SEEDS = [0]
 
SPLITS = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
}
BASE = "hf://datasets/stanfordnlp/sst2/"

def download_pretrained(option):
    if option == 'glove':
        destination = "glove.2024.wikigiga.50d.zip"
        url = "https://surfdrive.surf.nl/s/qYfZ6PiNrMLCwpS/download"
        dims = 50
        skip_first = False

    elif option == 'fasttext':
        destination = "wiki-news-300d-1M.vec.zip"
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
        dims = 300
        skip_first = True
    elif option == 'conceptnet':
        destination = "numberbatch-en-19.08.txt.gz"
        url = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
        dims = 300
        skip_first = True
    else:
        raise ValueError(f"Unknown option: {option}")

    if not Path(destination).is_file():
        print("Downloading...")
        urlretrieve(url, destination)
        print("Download complete!")

    return destination, dims, skip_first

def get_cv_splits(pairs):
    return [part1.experiment.cross_validation_split(pairs, k=NUM_FOLDS, seed=seed) for seed in SEEDS]

def to_pairs(df):
    return list(zip(df["sentence"], df["label"].astype(int)))

def load_sst2():
    train = to_pairs(pd.read_parquet(BASE + SPLITS["train"]))
    validation = to_pairs(pd.read_parquet(BASE + SPLITS["validation"]))
    test = to_pairs(pd.read_parquet(BASE + SPLITS["test"]))
    return train, validation, test



def get_embedding(word, model):
    index = model.vocab.index(word)
    return model.vectors[index]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def average_similarity(pairs, model):
    similarities = [
        cosine_similarity(get_embedding(w1, model), get_embedding(w2, model))
        for w1, w2 in pairs
    ]
    return np.mean(similarities)

class MeanPoolingEncoder:
    def __init__(self, tokenizer, embedding_model):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.output_size = embedding_model.vectors.shape[-1]

        self.word_to_idx = {word: i for i, word in enumerate(embedding_model.vocab)}

    def __call__(self, sentence):
        tokens = [
            token.lower()
            for token in self.tokenizer.tokenize(sentence)
        ]

        vectors = [
            self.embedding_model.vectors[self.word_to_idx[token]]
            for token in tokens
            if token in self.word_to_idx
        ]

        if not vectors:
            return np.zeros(self.output_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def __len__(self):
        return self.output_size