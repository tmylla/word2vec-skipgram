import pickle
import logging

from torch.utils.data import DataLoader
from dataloader import TrainDataset
from trainer import *
from utils import *

torch.manual_seed(0)


def set_logger():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='log/train.log',
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    set_logger()

    language = 'zh'
    MAX_VOCAB_SIZE = 50000
    raw_data = 'data/{}.txt'.format(language)
    text, word_dict, word2id, word_freqs = data_preprocess(raw_data, MAX_VOCAB_SIZE)
    with open('data/{}_word2id.pkl'.format(language), 'wb') as f:
        pickle.dump(word2id, f)
    logging.info('*' * 45)
    logging.info('Num_text : {}, Num_word : {}'.format(len(text), len(word2id.keys())))

    neg_sample = True
    vocab_dim = len(word2id.keys())
    embed_dim = 300
    C = 3
    K = 15
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.025

    train_dataset = TrainDataset(neg_sample, text, word2id, word_freqs, C, K)
    dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    model = Word2Vec(neg_sample, vocab_dim, embed_dim)
    logging.info('*' * 45)
    logging.info('Neg_sample : {}, Embed_dim : {}, Win_size : {}'.format(neg_sample, embed_dim, C))
    model.train(dataloader, num_epochs, learning_rate)
    torch.save(model.net.state_dict(), 'model/{}_word_embedding-{}-{}.pt'.format(language, neg_sample, embed_dim))
