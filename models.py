import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SequenceTagger(nn.Module):

    def __init__(self, embedding_dim, recurrent_dim, encoder_rec_layers, vocab_size, tagset_size):
        super(SequenceTagger, self).__init__()
        self.recurrent_dim=recurrent_dim
        self.word_embeddings=nn.Embedding(vocab_size, embedding_dim)

        self.recurrent=nn.LSTM(embedding_dim, self.recurrent_dim,bidirectional=True, num_layers=encoder_rec_layers)

        self.linear=nn.Linear(self.recurrent_dim*2, tagset_size)


    def forward(self, batch_seq):
        embeddings=self.word_embeddings(batch_seq)
        recurrent_out, hidden=self.recurrent(embeddings.view(len(batch_seq), batch_seq.size(1), -1))
        linear_out=self.linear(recurrent_out.view(len(batch_seq),batch_seq.size(1),-1))
        softmax_out=F.log_softmax(linear_out)
        return softmax_out
