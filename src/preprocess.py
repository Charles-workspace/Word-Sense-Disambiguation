import re
import pickle
from pathlib import Path
from collections import defaultdict

import re
import pickle
from pathlib import Path
from collections import defaultdict


TEXT_REGEX = re.compile(r'<TEXT[^>]*>(.*?)</TEXT>', re.DOTALL | re.IGNORECASE)
SENT_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')
TOKEN_REGEX = re.compile(r'\b[a-zA-Z]+\b')


def read_files(input_dir):
    for file_path in Path(input_dir).glob("*"):
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            yield f.read()


def extract_texts(raw_content):
    return TEXT_REGEX.findall(raw_content)


def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text) # Remove any remaining HTML tags
    text = text.replace("|", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_text(cleaned_text):
    sentences = SENT_SPLIT_REGEX.split(cleaned_text)

    tokenized = []
    for sent in sentences:
        tokens = TOKEN_REGEX.findall(sent.lower())
        if tokens:
            tokenized.append(tokens)

    return tokenized


def build_corpus(input_dir):
    corpus = []

    for raw in read_files(input_dir):
        for block in extract_texts(raw):
            cleaned = clean_text(block)
            tokenized = tokenize_text(cleaned)
            if tokenized:
                corpus.append(tokenized)

    return corpus


def build_token_index(corpus):
    index = defaultdict(list)

    for doc_id, document in enumerate(corpus):
        for sent_id, sentence in enumerate(document):
            for token_id, token in enumerate(sentence):
                index[token].append((doc_id, sent_id, token_id))

    return index


def get_context(corpus, doc_id, sent_id, token_id, window_size=2):
    sentence = corpus[doc_id][sent_id]
    start = max(0, token_id - window_size)
    end = min(len(sentence), token_id + window_size + 1)
    left = sentence[start:token_id]
    right = sentence[token_id+1:end]
    return left, right


def save_preprocessed_data(corpus, token_index, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)

    with open(out / 'token_index.pkl', 'wb') as f:
        pickle.dump(token_index, f)


def load_preprocessed_data(out_dir):
    out = Path(out_dir)

    with open(out / 'corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)

    with open(out / 'token_index.pkl', 'rb') as f:
        token_index = pickle.load(f)

    return corpus, token_index


if __name__ == "__main__":
    corpus_dir = 'Corpus-spell-AP88'
    out_dir = "data/preprocessed"

    print("Building corpus...")
    corpus = build_corpus(corpus_dir)
    print(f"Total documents extracted: {len(corpus)}")

    print("Building token index...")
    token_index = build_token_index(corpus)
    print(f"Unique tokens: {len(token_index)}")

    print("Saving preprocessed data...")
    save_preprocessed_data(corpus, token_index, out_dir)

    print("Done.")