import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SequenceTagger(nn.Module):

    def __init__(self, word_vocab_size, char_vocab_size, tagset_size, args, pretrained_size=None):
        super(SequenceTagger, self).__init__()
        self.recurrent_dim=args.recurrent_size
        self.char_embeddings=nn.Embedding(char_vocab_size, args.char_embedding_size, padding_idx=0)
        self.word_embeddings=nn.Embedding(word_vocab_size, args.word_embedding_size, padding_idx=0) # sparse=True

        recurrent_input_dim=args.word_embedding_size+args.char_recurrent_size*2 # *2 because of bidirectional

        if len(args.pretrained_word_embeddings)>0:
            self.pretrained_word_embeddings=nn.Embedding(word_vocab_size, pretrained_size, padding_idx=0)
            self.pretrained_word_embeddings.weight.requires_grad = False
            recurrent_input_dim+=pretrained_size
        else:
            self.pretrained_word_embeddings=None

        self.char_lstm=nn.LSTM(args.char_embedding_size, args.char_recurrent_size, bidirectional=True, num_layers=1, dropout=args.recurrent_dropout)

#        self.chars2word_lstm=nn.LSTM(input_size=self.char_lstm.hidden_size*self.char_lstm.num_layers*2, hidden_size=args.recurrent_dim, num_layers=1, bidirectional=True)


        self.recurrent=nn.LSTM(recurrent_input_dim, self.recurrent_dim, bidirectional=True, num_layers=args.encoder_layers, dropout=args.recurrent_dropout)

        self.linear=nn.Linear(self.recurrent_dim*2, tagset_size)




    def forward(self, word_batch_seq, char_batch_seq, sequence_lenghts):

        wcount,mbatch,ccount=char_batch_seq.size() # word X sentence X character
        _char_lstm_in=char_batch_seq.view(wcount*mbatch,ccount).transpose(0,1).contiguous()
        _char_embeddings=self.char_embeddings(_char_lstm_in)

        _,(chr_h_n,_)=self.char_lstm(_char_embeddings)
        chr_h_n_wrd_input=chr_h_n.transpose(0,1).contiguous().view(wcount,mbatch,-1)

        _word_embeddings=self.word_embeddings(word_batch_seq)

        if self.pretrained_word_embeddings:
            _pretrained_word_embeddings=self.pretrained_word_embeddings(word_batch_seq)
            chr_h_n_wrd_input_concat=torch.cat((_pretrained_word_embeddings, _word_embeddings, chr_h_n_wrd_input), dim=2)
        else:
            chr_h_n_wrd_input_concat=torch.cat((_word_embeddings, chr_h_n_wrd_input), dim=2)

#        chr_h_n_wrd_input_sum=(chr_h_n_wrd_input+_word_embeddings)/2

        # pack
#        print(list(sequence_lenghts))
#        print("without view:",chr_h_n_wrd_input_concat.size())
#        print("with view:",chr_h_n_wrd_input_concat.view(len(word_batch_seq), word_batch_seq.size(1), -1).size())
        chr_h_n_wrd_input_concat_packed=torch.nn.utils.rnn.pack_padded_sequence(chr_h_n_wrd_input_concat, list(sequence_lenghts))
#        print(chr_h_n_wrd_input_concat_packed)
        recurrent_out, self.hidden=self.recurrent(chr_h_n_wrd_input_concat_packed)#.view(len(word_batch_seq), word_batch_seq.size(1), -1))
#        print(recurrent_out)
        linear_out=self.linear(recurrent_out.data)#.view(len(word_batch_seq),word_batch_seq.size(1),-1))
        newly_packed = torch.nn.utils.rnn.PackedSequence(linear_out, recurrent_out.batch_sizes)  # create PackedSequence

#        softmax_out=F.log_softmax(linear_out,dim=2) # or log_softmax

        return newly_packed
