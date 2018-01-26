import sys
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

from data import POSdata, MorphoData, TorchData
from models import SequenceTagger

from text_classifier_torch.t2i import T2I as text_vectorizer
from text_classifier_torch.t2i import to_torch_long_tensor,torch_minibatched_2dim

WARNING='\033[91m'
END_WARNING='\033[0m'

def predictions2text(predictions,label_vectorizer):
    text_labels=[]
    # unpack because predictions is packed sequence
    data, lens = torch.nn.utils.rnn.pad_packed_sequence(predictions)  # unpack
#    print("*******",data.size())
    scores,predictions_=data.max(2)
    for pred in torch.transpose(predictions_,0,1):
        max_labels=[label_vectorizer.reverse(int(i)) for i in pred.data.cpu().numpy()]
        text_labels.append(max_labels)
    return text_labels


def accuracy(data,model,label_vectorizer,targets,args,sentences,verbose=False):
    correct=0
    total=0
    _word_in,_char_in,_lengths=data
    number_of_batches=_word_in.size(1)
    for batch_id in range(number_of_batches):
        word_batch=autograd.Variable(_word_in[:,batch_id,:])
        char_batch=autograd.Variable(_char_in[:,batch_id,:])
        len_batch=torch.LongTensor(_lengths[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size])
        if args.cuda:
            word_batch=word_batch.cuda()
            char_batch=char_batch.cuda()
        model.eval()
        predictions=predictions2text(model(word_batch,char_batch,len_batch),label_vectorizer)
        target_batch=targets[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size]
        sentence_batch=sentences[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size]
        for psent,sent,sentence in zip(predictions,target_batch,sentence_batch):
            if verbose:
                print("PRED:",psent)
                print("GOLD:",sent)
                print("INPUT:",sentence,"\n")
            for ptoken,token in zip(psent,sent):
                if token=="<START>" or token=="<END>":
                    continue
                if token=="__PADDING__":
                    break
                if token==ptoken:
                    correct+=1
                total+=1
    return correct/total*100 if total!= 0 else 0

def minibatched_3dim(data,batch_size):
    seq_count,word_count,char_count=data.size()
    seq_count_mbatch_aligned=(seq_count//batch_size)*batch_size
    data_batched=data[:seq_count_mbatch_aligned].transpose(0,1).contiguous().view(word_count,seq_count//batch_size,-1,char_count)
    return data_batched


def train(args):

    # data
#    posdata=POSdata() # class to read part-of-speech tagging data from conllu
    data_reader=MorphoData() # class to read part-of-speech tagging data from conllu
    train_sentences,train_labels=data_reader.read_data(args.train_file, args.max_train, count_words=False)
#    train_sentences=posdata.mask_rare_words(train_sentences, freq=args.word_freq_cutoff, mask_term="__UNK__", verbose=args.verbose)
    print("First training example:", train_sentences[0], train_labels[0])

    torchdata=TorchData() # class to turn text into torch style minibatches
    if len(args.pretrained_word_embeddings)>0:
        torchdata.init_vocab_from_pretrained(args.pretrained_word_embeddings)
    
    train_batches_word, train_batches_char, train_batches_label, train_sequence_lengths, train_sorting_indices, train_unsorting_indices=torchdata.prepare_torch_data(train_sentences, train_labels, args.batch_size, args.max_seq_len, args.max_seq_len_char, train=True, shuffle=args.shuffle_train, sort_batch=True) # sentences, labels, batch_size, seq_words, seq_chars, train=True, shuffle=False, sort_batch=False

    print("Train word input sizes:",train_batches_word.size())
    print("Train character input sizes:",train_batches_char.size())
    print("Train target sizes:",train_batches_label.size())
    print("Training examples:",len(train_sentences),"Unique words:",len(torchdata.word_vectorizer.idict),"Unique characters:",len(torchdata.char_vectorizer.idict),"Classes:",len(torchdata.label_vectorizer.idict))

    

    # model
    model=SequenceTagger(len(torchdata.word_vectorizer.idict), len(torchdata.char_vectorizer.idict), len(torchdata.label_vectorizer.idict), args, pretrained_size=torchdata.embedding_size)

    # copy weights from pretrained embeddings
    if len(args.pretrained_word_embeddings)>0:
        torchdata.load_pretrained_embeddings(args.pretrained_word_embeddings, model, model.pretrained_word_embeddings)

    if args.cuda:
        model.cuda()
    
    # remove frozen parameters
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    loss_function=nn.CrossEntropyLoss()#ignore_index=0)
    optimizer=optim.SGD(trainable_parameters,lr=args.learning_rate)
#    optimizer=optim.Adam(trainable_parameters,lr=args.learning_rate)

    number_of_batches=train_batches_word.size(1)
    
    
    print("Training batches",number_of_batches)
    print("Shuffling training data:",args.shuffle_train)

    # devel data
    devel_sentences,devel_labels=data_reader.read_data(args.devel_file,args.max_devel)

    devel_batches_word, devel_batches_char, devel_batches_label, devel_sequence_lengths, devel_sorting_indices, devel_unsorting_indices=torchdata.prepare_torch_data(devel_sentences, devel_labels, args.batch_size, args.max_seq_len, args.max_seq_len_char, train=False, shuffle=False, sort_batch=True)
    devel_sentences_sorted=list(np.array(devel_sentences)[devel_sorting_indices])
    devel_labels_sorted=list(np.array(devel_labels)[devel_sorting_indices])

    print("Devel word input sizes:",devel_batches_word.size())
    print("Devel character input sizes:",devel_batches_char.size())
    print("Devel target sizes:",devel_batches_label.size())

    # TRAINING
    for epoch in range(args.epochs):
        if epoch is not 0:
            acc=accuracy((devel_batches_word, devel_batches_char, devel_sequence_lengths), model, torchdata.label_vectorizer, devel_labels_sorted, args, devel_sentences_sorted, args.verbose)
            print("EPOCH:", epoch, "LOSS:", loss.data[0], "ACCURACY:", acc, flush=True)

        # shuffle batches
        idxs=[i for i in range(number_of_batches)]
        np.random.shuffle(idxs)


        # shuffle training data (...and create new batches)
        if args.shuffle_train:
            train_batches_word, train_batches_char, train_batches_label, train_sequence_lengths, train_sorting_indices, train_unsorting_indices=torchdata.prepare_torch_data(train_sentences, train_labels, args.batch_size, args.max_seq_len, args.max_seq_len_char, train=True, shuffle=True, sort_batch=True)


#        for batch_id in range(number_of_batches):
        for batch_id in idxs:

            word_batch=autograd.Variable(train_batches_word[:,batch_id,:])
            char_batch=autograd.Variable(train_batches_char[:,batch_id,:])
            targets=autograd.Variable(train_batches_label[:,batch_id,:])
            lengths=torch.LongTensor(train_sequence_lengths[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size])
    
            if args.cuda:
                word_batch=word_batch.cuda()
                char_batch=char_batch.cuda()
                targets=targets.cuda()
                lengths=lengths.cuda()

            optimizer.zero_grad()
            model.train()
            outputs=model(word_batch, char_batch, lengths)

            # pack targets
            targets_packed=torch.nn.utils.rnn.pack_padded_sequence(targets, list(lengths))

#            loss=loss_function(outputs.contiguous().view(outputs.size(0)*outputs.size(1),-1), targets_packed.data.contiguous().view(outputs.size(0)*outputs.size(1)))
            loss=loss_function(outputs.data, targets_packed.data)

            loss.backward()
            optimizer.step()
    acc=accuracy((devel_batches_word, devel_batches_char, devel_sequence_lengths), model, torchdata.label_vectorizer, devel_labels_sorted, args, devel_sentences_sorted, args.verbose)
    print("EPOCH:", epoch, "LOSS:", loss.data[0], "ACCURACY:", acc, flush=True)

    if len(args.save_directory)>0:
        print("Saving model into",args.save_directory)

        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
        print("Model:",os.path.join(args.save_directory,"model.pt"))
        torch.save(model, os.path.join(args.save_directory,"model.pt"))
        print("Vectorizers:",os.path.join(args.save_directory,"torchdata_vectorizers.pkl"))
        with open(os.path.join(args.save_directory,"torchdata_vectorizers.pkl"),"wb") as f:
            pickle.dump(torchdata,f)
        print("Saving ready.")
        
    return acc



def callable(inputs):
    # inputs is a list of hyperparameters (batch_size, word_embedding_size, char_embedding_size, recurrent_size, recurrent_dropout, learning_rate)
    # returns final accuracy

    from collections import namedtuple
    Arguments = namedtuple('Arguments', ['train_file', 'devel_file', 'max_train', 'max_devel', 'cuda', 'verbose', 'shuffle_train', 'epochs', 'encoder_layers', 'max_seq_len', 'max_seq_len_char', 'word_freq_cutoff', 'pretrained_word_embeddings', 'batch_size', 'word_embedding_size', 'char_embedding_size', 'recurrent_size', 'recurrent_dropout', 'learning_rate'])

    args=Arguments("../UD_Finnish/fi-ud-train.conllu", "../UD_Finnish/fi-ud-dev.conllu", 1000000, 1000000, True, False, True, 10, 2, 50, 20, 5, "", int(inputs[0])**2, int(inputs[1])*100, int(inputs[2])*100, int(inputs[3])*100, float(inputs[4])/10, float(inputs[5])/10000)

    print(args)

    accuracy=train(args)

    return 100-accuracy
    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('--train_file', type=str, help='Input training file name')
    g.add_argument('--devel_file', type=str, help='Input development file name')
    g.add_argument('--max_train', type=int, default=1000000, help='Maximum number of sentences used in training')
    g.add_argument('--max_devel', type=int, default=1000000, help='Maximum number of sentences used in training eval')
    g.add_argument('--cpu', dest='cuda', default=True, action="store_false", help='Use cpu.')
    g.add_argument('--batch_size', type=int, default=64, help='Minibatch size')
    g.add_argument('--word_embedding_size', type=int, default=200, help='Size of word embeddings')
    g.add_argument('--char_embedding_size', type=int, default=200, help='Size of word embeddings')
    g.add_argument('--recurrent_size', type=int, default=500, help='Size of recurrent layers')
    g.add_argument('--char_recurrent_size', type=int, default=300, help='Size of recurrent layers in character rnn')
    g.add_argument('--encoder_layers', type=int, default=2, help='Number of recurrent layer in the endocer')
    g.add_argument('--recurrent_dropout', type=float, default=0.0, help='Dropout in the recurrent layers')
    g.add_argument('--max_seq_len', type=int, default=100, help='Max sentence len (words in sentence)')
    g.add_argument('--max_seq_len_char', type=int, default=30, help='Max word len (characters in word)')
    g.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    g.add_argument('--shuffle_train', default=False, action='store_true', help='Shuffle training data between epochs')
    g.add_argument('--word_freq_cutoff', type=int, default=2, help='Cutoff frequency for words, use unknown in training if less than this')
    g.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    g.add_argument('--pretrained_word_embeddings', type=str, default="", help='Pretrained word embeddings file (.bin or .vectors)')
    g.add_argument('--verbose', default=False, action='store_true', help='Verbose prints during training')
    g.add_argument('--save_directory', type=str, default="", help='Directory to save the model and vectorizers. Model will be .pt file and vectorizer .pkl files. Default="" --> do not save.')
    
    args = parser.parse_args()

    if len(args.save_directory)==0:
        print(WARNING, "Warning! Model will not be saved, use --save_directory argument to save the model into a directory.", END_WARNING)
    if os.path.exists(args.save_directory):
        print(WARNING, "Warning! Model saving directory already exists, old files may be overwritten when the training ends.", END_WARNING)

    accuracy=train(args)
