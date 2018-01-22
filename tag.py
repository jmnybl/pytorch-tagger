import sys
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import shuffle
import pickle
from itertools import zip_longest

from data import POSdata, TorchData
from models import SequenceTagger
from train import accuracy

from text_classifier_torch.t2i import T2I as text_vectorizer
from text_classifier_torch.t2i import to_torch_long_tensor,torch_minibatched_2dim


WARNING='\033[91m'
END_WARNING='\033[0m'

def predictions2text(predictions,label_vectorizer):
    text_labels=[]
    scores,predictions_=predictions.max(2)
    for pred in torch.transpose(predictions_,0,1):
        max_labels=[label_vectorizer.reverse(int(i)) for i in pred.data.cpu().numpy()]
        text_labels.append(max_labels)
    return text_labels

ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)
#def predict(data,model,label_vectorizer,targets,args,sentences,verbose=False):
def predict(input_data, model, label_vectorizer, conllu_sentences):
    _word_in,_char_in=input_data
    number_of_batches=_word_in.size(1)
    for batch_id in range(number_of_batches):
        word_batch=autograd.Variable(_word_in[:,batch_id,:])
        char_batch=autograd.Variable(_char_in[:,batch_id,:])
        if args.cuda:
            word_batch=word_batch.cuda()
            char_batch=char_batch.cuda()
        model.eval()
        predictions=predictions2text(model(word_batch,char_batch),label_vectorizer)
        conllu_batch=conllu_sentences[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size]
        for psent,(comm,conllu) in zip_longest(predictions,conllu_batch,fillvalue=[]): # this is one sequence, i.e. sentence
            if not conllu:
                #print
                break
            sentence=[]
            for i, token in enumerate(conllu):
                ptoken=psent[i+1] # remove start marker to keep aligned
                if ptoken=="<START>" or ptoken=="<END>" or ptoken=="__PADDING__":
                    ptoken="_"
                token[UPOS]=ptoken
                sentence.append(token)
            for c in comm:
                print(c)
            for token in sentence:
                print("\t".join(t for t in token))
            print("")





def predict_eval(args):

    # data
    posdata=POSdata()
    input_sentences,input_labels=posdata.read_data(args.input_file, 100000000, count_words=False)
    conllu_sentences=[(c,s) for c, s in posdata.conllu_reader(open(args.input_file,"rt"))]

    with open(os.path.join(args.model_dir,"torchdata_vectorizers.pkl"),"rb") as f:
        torchdata=pickle.load(f) # class to turn text into torch style minibatches
    
    # TODO: max sequence lengths? Should collect from the data?
    data_batches_word, data_batches_char, data_batches_label=torchdata.prepare_torch_data(input_sentences, input_labels, args.batch_size, args.max_seq_len, args.max_seq_len_char, train=False, shuffle=False) # sentences, labels, batch_size, seq_words, seq_chars, train=True, shuffle=False



    model=torch.load(os.path.join(args.model_dir,"model.pt"))

    if args.cuda:
        model.cuda()
    
    if args.evaluate:
        acc=accuracy((data_batches_word, data_batches_char), model, torchdata.label_vectorizer, input_labels, args, input_sentences, args.verbose)
        print("ACCURACY:",acc)
        
    else:
        predict((data_batches_word, data_batches_char), model, torchdata.label_vectorizer, conllu_sentences)





if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('--input_file', type=str, required=True, help='Input file name')
    g.add_argument('--model_dir', type=str, required=True, help='Model directory name with pytorch models (.pt) and vectorizers (.pkl)')
    g.add_argument('--evaluate', default=False, action='store_true', help='Evaluate model, do not output predictions.')
    g.add_argument('--cpu', dest='cuda', default=True, action="store_false", help='Use cpu (default False).')
    g.add_argument('--batch_size', type=int, default=64, help='Minibatch size')
    g.add_argument('--max_seq_len', type=int, default=100, help='Max sentence len (words in sentence)')
    g.add_argument('--max_seq_len_char', type=int, default=30, help='Max word len (characters in word)')
    g.add_argument('--verbose', default=False, action='store_true', help='Verbose prints')
    
    args = parser.parse_args()

    if args.batch_size!=1:
        print(WARNING, "Batching does not work properly right now, forcing batch_size to one temporarily.", END_WARNING, file=sys.stderr)
        args.batch_size=1

    predict_eval(args)
