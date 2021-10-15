import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from model import SkipGram, SkipGram_neg

torch.manual_seed(0)


class Word2Vec():
    def __init__(self, neg_sample, vocab_dim, embed_dim):
        self.neg_sample = neg_sample
        self.embed_dim = embed_dim
        self.early_stop = EarlyStopping()
        if not self.neg_sample:
            self.net = SkipGram(vocab_dim, embed_dim)
        else:
            self.net = SkipGram_neg(vocab_dim, embed_dim)
        
    def train(self, dataloader, num_epochs, learning_rate):
        device = torch.device('cuda:0')
        self.net.to(device)

        if not self.neg_sample:
            loss_function = nn.NLLLoss()
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/2)

            for epoch in range(num_epochs):
                losses = []
                for i, (cen_tensor, pos_tensors, neg_tensors) in enumerate(dataloader):
                    for pos_tensor in pos_tensors.T:
                        cen_tensor, pos_tensor = cen_tensor.to(device), pos_tensor.to(device)
                        optimizer.zero_grad()
                        loss = loss_function(self.net.forward(cen_tensor), pos_tensor)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.detach().cpu().numpy())
                    if i%300 == 0:
                        logging.info('Epoch : {}, Iteration : {}, Loss : {:.6f}'.format(epoch, i, loss.item()))
                torch.save(self.net.state_dict(), 'model/word_embedding_tmp.pt')
                self.early_stop.update_loss(np.mean(losses))
                if self.early_stop.stop_training():
                    logging.info('*' * 45)
                    print(self.early_stop.loss_list)
                    break
                scheduler.step()
        elif self.neg_sample:
            # loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/2)

            for epoch in range(num_epochs):
                losses = []
                for i, (cen_tensor, pos_tensors, neg_tensors) in enumerate(dataloader):
                    cen_tensor, pos_tensors, neg_tensors = cen_tensor.to(device), pos_tensors.to(device), neg_tensors.to(device)
                    optimizer.zero_grad()
                    loss = self.net.forward(cen_tensor, pos_tensors, neg_tensors).mean()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach().cpu().numpy())
                    if i%300 == 0:
                        logging.info('Epoch : {}, Iteration : {}, Loss : {:.6f}'.format(epoch, i, loss.item()))
                torch.save(self.net.state_dict(), 'model/word_embedding_tmp.pt')
                self.early_stop.update_loss(np.mean(losses))
                if self.early_stop.stop_training():
                    logging.info('*' * 45)
                    print(self.early_stop.loss_list)
                    break
                scheduler.step()


class EarlyStopping():
    def __init__(self, patience=3, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        logging.info("Loss gain: {}%".format(round(100 * gain, 2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False