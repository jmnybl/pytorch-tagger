import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import shuffle

from data import POSdata, TorchData
from models import SequenceTagger

from text_classifier_torch.t2i import T2I as text_vectorizer
from text_classifier_torch.t2i import to_torch_long_tensor,torch_minibatched_2dim


def predictions2text(predictions,label_vectorizer):
    text_labels=[]
    scores,predictions_=predictions.max(2)
    for pred in torch.transpose(predictions_,0,1):
        max_labels=[label_vectorizer.reverse(int(i)) for i in pred.data.cpu().numpy()]
        text_labels.append(max_labels)
    return text_labels


def accuracy(data,model,label_vectorizer,targets,args,sentences,verbose=False):
    correct=0
    total=0
    _word_in,_char_in=data
    number_of_batches=_word_in.size(1)
    for batch_id in range(number_of_batches):
        word_batch=autograd.Variable(_word_in[:,batch_id,:])
        char_batch=autograd.Variable(_char_in[:,batch_id,:])
        if args.cuda:
            word_batch=word_batch.cuda()
            char_batch=char_batch.cuda()
        model.eval()
        predictions=predictions2text(model(word_batch,char_batch),label_vectorizer)
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
    posdata=POSdata()
    train_sentences,train_labels=posdata.read_data(args.train_file, args.max_train, count_words=True)
    train_sentences=posdata.mask_rare_words(train_sentences, freq=args.word_freq_cutoff, mask_term="__UNK__", verbose=args.verbose)

    torchdata=TorchData() # class to turn text into torch style minibatches
    
    train_batches_word, train_batches_char, train_batches_label=torchdata.prepare_torch_data(train_sentences, train_labels, args.batch_size, args.max_seq_len, args.max_seq_len_char, train=True, shuffle=args.shuffle_train) # sentences, labels, batch_size, seq_words, seq_chars, train=True, shuffle=False

    print("Train word input sizes:",train_batches_word.size())
    print("Train character input sizes:",train_batches_char.size())
    print("Train target sizes:",train_batches_label.size())
    print("Training examples:",len(train_sentences),"Unique words:",len(torchdata.word_vectorizer.idict),"Unique characters:",len(torchdata.char_vectorizer.idict),"Classes:",len(torchdata.label_vectorizer.idict))

    

    vocabulary_size=torchdata.calculate_vocabulary_size(args) # expand the vocabulary from pretrained embeddings if needed
    print("Expanded word vocabulary size:", vocabulary_size)

    # model
    model=SequenceTagger(vocabulary_size,len(torchdata.char_vectorizer.idict),len(torchdata.label_vectorizer.idict),args)

    if len(args.pretrained_word_embeddings)>0:
        torchdata.load_pretrained_embeddings(args.pretrained_word_embeddings, torchdata.word_vectorizer, model, model.word_embeddings)

    if args.cuda:
        model.cuda()
    
    loss_function=nn.CrossEntropyLoss()
#    optimizer=optim.SGD(model.parameters(),lr=args.learning_rate)
    optimizer=optim.Adam(model.parameters(),lr=args.learning_rate)

    number_of_batches=train_batches_word.size(1)
    
    
    print("Training batches",number_of_batches)
    print("Shuffling training data:",args.shuffle_train)

    # devel data
    devel_sentences,devel_labels=posdata.read_data(args.devel_file,args.max_devel)

    devel_batches_word, devel_batches_char, devel_batches_label=torchdata.prepare_torch_data(devel_sentences, devel_labels, args.batch_size, args.max_seq_len, args.max_seq_len_char, train=False, shuffle=False)

    print("Devel word input sizes:",devel_batches_word.size())
    print("Devel character input sizes:",devel_batches_char.size())
    print("Devel target sizes:",devel_batches_label.size())

    # now we can start training
    for epoch in range(args.epochs):
        if epoch is not 0:
            acc=accuracy((devel_batches_word, devel_batches_char), model, torchdata.label_vectorizer, devel_labels, args, devel_sentences, args.verbose)
            print("EPOCH:", epoch, "LOSS:", loss.data[0], "ACCURACY:", acc, flush=True)

#        # shuffle batches
#        idxs=[i for i in range(number_of_batches)]
#        shuffle(idxs)


        # shuffle training data (...and create new batches)
        if args.shuffle_train:
            train_batches_word, train_batches_char, train_batches_label=torchdata.prepare_torch_data(train_sentences, train_labels, args.batch_size, args.max_seq_len, args.max_seq_len_char, train=True, shuffle=True)


        for batch_id in range(number_of_batches):

            word_batch=autograd.Variable(train_batches_word[:,batch_id,:])
            char_batch=autograd.Variable(train_batches_char[:,batch_id,:])
            targets=autograd.Variable(train_batches_label[:,batch_id,:])
    
            if args.cuda:
                word_batch=word_batch.cuda()
                char_batch=char_batch.cuda()
                targets=targets.cuda()

            optimizer.zero_grad()
            model.train()
            outputs=model(word_batch,char_batch)

            loss=loss_function(outputs.contiguous().view(outputs.size(0)*outputs.size(1),-1), targets.contiguous().view(outputs.size(0)*outputs.size(1)))

            loss.backward()
            optimizer.step()
    return acc

def callable(inputs):
    # inputs is a list of hyperparameters (batch_size, word_embedding_size, char_embedding_size, recurrent_size, recurrent_dropout, learning_rate)
    # returns final accuracy

    from collections import namedtuple
    Arguments = namedtuple('Arguments', ['train_file', 'devel_file', 'max_train', 'max_devel', 'cuda', 'verbose', 'shuffle_train', 'epochs', 'encoder_layers', 'max_seq_len', 'max_seq_len_char', 'word_freq_cutoff', 'pretrained_word_embeddings', 'batch_size', 'word_embedding_size', 'char_embedding_size', 'recurrent_size', 'recurrent_dropout', 'learning_rate'])

    args=Arguments("../UD_Finnish/fi-ud-train.conllu", "../UD_Finnish/fi-ud-dev.conllu", 1000000, 1000000, True, False, True, 10, 2, 50, 20, 5, "", int(inputs[0])**2, int(inputs[1])*100, int(inputs[2])*100, int(inputs[3])*100, float(inputs[4])/10, float(inputs[5])/10000)

    print(args)

    accuracy=train(args)

    return accuracy*-1
    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('--train_file', type=str, help='Input training file name')
    g.add_argument('--devel_file', type=str, help='Input development file name')
    g.add_argument('--max_train', type=int, default=5000, help='Maximum number of sentences used in training')
    g.add_argument('--max_devel', type=int, default=100, help='Maximum number of sentences used in training')
    g.add_argument('--cpu', dest='cuda', default=True, action="store_false", help='Use cpu.')
    g.add_argument('--batch_size', type=int, default=64, help='Minibatch size')
    g.add_argument('--word_embedding_size', type=int, default=200, help='Size of word embeddings')
    g.add_argument('--char_embedding_size', type=int, default=200, help='Size of word embeddings')
    g.add_argument('--recurrent_size', type=int, default=500, help='Size of recurrent layers')
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
    
    args = parser.parse_args()

    accuracy=train(args)
