import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import POSdata
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


def accuracy(data,model,label_vectorizer,targets):
    correct=0
    total=0
    predictions=predictions2text(model(data),label_vectorizer)
    for psent,sent in zip(predictions,targets):
        for ptoken,token in zip(psent,sent):
            if token=="__PADDING__":
                break
            if token==ptoken:
                correct+=1
            total+=1
    return correct/total*100 if total!= 0 else 0

    

def train(args):

    # data
    posdata=POSdata()
    train_sentences,train_labels_text=posdata.read_data(args.train_file,args.max_train)
    print(train_sentences[0],train_labels_text[0])
    print(len(train_sentences),len(train_labels_text))
    
    vectorizer_data=text_vectorizer()
    vectorizer_labels=text_vectorizer(with_unknown=None)
    
    train_data=vectorizer_data(train_sentences)
    train_labels=vectorizer_labels(train_labels_text)
    print(train_data[0],train_labels[0])
    print(len(train_data),len(train_labels))

    train_batches=torch_minibatched_2dim(to_torch_long_tensor(train_data,args.max_seq_len),args.batch_size)
    label_batches=torch_minibatched_2dim(to_torch_long_tensor(train_labels,args.max_seq_len),args.batch_size)
    print("Train sizes:",train_batches.size(),label_batches.size())

    devel_sentences,devel_labels_text=posdata.read_data(args.devel_file,args.max_devel)
    devel_batches=torch_minibatched_2dim(to_torch_long_tensor(vectorizer_data(devel_sentences),args.max_seq_len),args.batch_size)
    devel_labels=torch_minibatched_2dim(to_torch_long_tensor(vectorizer_labels(devel_labels_text),args.max_seq_len),args.batch_size)
    print("Devel sizes:",devel_batches.size(),devel_labels.size())
    
    devel_batches=autograd.Variable(devel_batches[:,0,:])

    # model
    model=SequenceTagger(args.word_embedding_size, args.recurrent_size, args.encoder_layers, len(vectorizer_data.idict),len(vectorizer_labels.idict))

    if args.cuda:
        model.cuda()
        devel_batches=devel_batches.cuda()

    # now we can start training
    loss_function=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=1)

    number_of_batches=train_batches.size(1)

#    predictions_before=model(devel_batches)
    
    
    print("Training batches",number_of_batches)

    for epoch in range(args.epochs):
        if epoch is not 0:
            print("EPOCH:",epoch, "LOSS:",loss.data[0], "ACCURACY:",accuracy(devel_batches,model,vectorizer_labels,devel_labels_text[:args.batch_size]))

        for batch_id in range(number_of_batches):
            #print("BATCH:",batch_id)

            batch=autograd.Variable(train_batches[:,batch_id,:])
            targets=autograd.Variable(label_batches[:,batch_id,:])
    
            if args.cuda:
                batch=batch.cuda()
                targets=targets.cuda()

            optimizer.zero_grad()
            outputs=model(batch)
            loss=loss_function(outputs.contiguous().view(outputs.size(0)*outputs.size(1),-1), targets.contiguous().view(outputs.size(0)*outputs.size(1)))
            loss.backward()
            optimizer.step()
#            print("LOSS:",loss)


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
    g.add_argument('--recurrent_size', type=int, default=500, help='Size of recurrent layers')
    g.add_argument('--encoder_layers', type=int, default=2, help='Number of recurrent layer in the endocer')
    g.add_argument('--max_seq_len', type=int, default=100, help='Max sequence len')
    g.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    
    args = parser.parse_args()

    train(args)
