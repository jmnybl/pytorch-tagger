import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SequenceTagger(nn.Module):

    def __init__(self, word_vocab_size, char_vocab_size, tagset_size, args):
        super(SequenceTagger, self).__init__()
        self.recurrent_dim=args.recurrent_size
        self.char_embeddings=nn.Embedding(char_vocab_size, args.char_embedding_size, padding_idx=0)
        self.word_embeddings=nn.Embedding(word_vocab_size, args.word_embedding_size, padding_idx=0) # sparse=True

        self.char_lstm=nn.LSTM(args.char_embedding_size, int(args.word_embedding_size/2), bidirectional=True, num_layers=1)

#        self.chars2word_lstm=nn.LSTM(input_size=self.char_lstm.hidden_size*self.char_lstm.num_layers*2, hidden_size=args.recurrent_dim, num_layers=1, bidirectional=True)

        self.recurrent=nn.LSTM(args.word_embedding_size, self.recurrent_dim, bidirectional=True, num_layers=args.encoder_layers, dropout=args.recurrent_dropout)

        self.linear=nn.Linear(self.recurrent_dim*2, tagset_size)




    def forward(self, word_batch_seq, char_batch_seq):

        wcount,mbatch,ccount=char_batch_seq.size() # word X sentence X character
        _char_lstm_in=char_batch_seq.view(wcount*mbatch,ccount).transpose(0,1).contiguous()
        _char_embeddings=self.char_embeddings(_char_lstm_in)
        _,(chr_h_n,_)=self.char_lstm(_char_embeddings)
        chr_h_n_wrd_input=chr_h_n.transpose(0,1).contiguous().view(wcount,mbatch,-1)

        _word_embeddings=self.word_embeddings(word_batch_seq)

        chr_h_n_wrd_input_sum=(chr_h_n_wrd_input+_word_embeddings)/2

        recurrent_out, self.hidden=self.recurrent(chr_h_n_wrd_input_sum.view(len(word_batch_seq), word_batch_seq.size(1), -1))
        linear_out=self.linear(recurrent_out.view(len(word_batch_seq),word_batch_seq.size(1),-1))

#        softmax_out=F.log_softmax(linear_out,dim=2) # or log_softmax

        return linear_out
