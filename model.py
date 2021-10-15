import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class SkipGram(nn.Module):
    def __init__(self, vocab_dim, embed_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_dim)

    def forward(self, cen_tensor):
        emb = self.embeddings(cen_tensor)
        hidden = self.linear(emb)
        out = F.log_softmax(hidden, dim=1)

        return out

class SkipGram_neg(nn.Module):
    def __init__(self, vocab_dim, embed_dim):
        super(SkipGram_neg, self).__init__()
        self.in_embed = nn.Embedding(vocab_dim, embed_dim)
        self.out_embed = nn.Embedding(vocab_dim, embed_dim)

    def forward(self, cen_tensor, pos_tensors, neg_tensors):
        ''' center words, [batch_size]
            positive words, [batch_size, (window_size * 2)]
            negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        cen_embedding = self.in_embed(cen_tensor)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_tensors)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_tensors)  # [batch_size, (window * 2 * K), embed_size]

        cen_embedding = cen_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]
        pos_dot = torch.bmm(pos_embedding, cen_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]
        log_pos = F.logsigmoid(pos_dot).sum(1)

        neg_dot = torch.bmm(neg_embedding, cen_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]
        log_neg = F.logsigmoid(-neg_dot).sum(1)

        loss = log_pos + log_neg
        return -loss