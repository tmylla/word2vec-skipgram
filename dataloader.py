import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, neg_sample, text, word2idx, word_freqs, C, K):
        # text: a list of words, all text from the training dataset
        super(TrainDataset, self).__init__()
        self.neg_sample = neg_sample
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)  # nn.Embedding需要传入LongTensor类型
        self.word_freqs = torch.Tensor(word_freqs)
        self.C = C
        self.K = K

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 返回以下数据用于训练：
            - 中心词
            - 这个单词附近的positive word
            - 如果neg_sample==True，返回随机采样的K个单词作为negative word
        '''
        center_words = self.text_encoded[idx]
        pos_indices = list(range(idx - self.C, idx)) + list(range(idx + 1, idx + self.C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words