import torch
import torch.nn as nn
import numpy as np
from collections import Counter

torch.manual_seed(0)


def data_preprocess(raw_data, MAX_VOCAB_SIZE):
    with open(raw_data, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    text = ' '.join(sentence.strip() for sentence in sentences).split()

    word_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
    word_dict['<UNK>'] = len(text) - sum(list(word_dict.values()))
    word2id = {word:i for i, word in enumerate(word_dict.keys())}

    word_counts = np.array([count for count in word_dict.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)

    return text, word_dict, word2id, word_freqs


def get_closest_word(word2id, word_embeddings, word, topn=10):
    index = word2id[word]
    embedding = word_embeddings[index]
    id2word = {id:word for word, id in word2id.items()}

    pdist = nn.PairwiseDistance()
    # TODO
    cos_dis = np.array([float(pdist(e.reshape(1,3), embedding.reshape(1,3))) for e in word_embeddings])
    return [id2word[i] for i in cos_dis.argsort()[:topn]]
