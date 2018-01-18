import torch
from numpy.random import shuffle

from text_classifier_torch.t2i import T2I as text_vectorizer
from text_classifier_torch.t2i import to_torch_long_tensor,torch_minibatched_2dim


ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)
class POSdata(object):

    def __init__(self):
        self.word_counts=None
        

    def conllu_reader(self,f):
        sent=[]
        comment=[]
        for line in f:
            line=line.strip()
            if not line: # new sentence
                if sent:
                    yield comment,sent
                comment=[]
                sent=[]
            elif line.startswith("#"):
                comment.append(line)
            else: #normal line
                sent.append(line.split("\t"))
        else:
            if sent:
                yield comment, sent


    def read_data(self, filename, max_sent, count_words=True):
        sentences=[]
        labels=[]
        if count_words:
            self.word_counts={}
        for comm, sent in self.conllu_reader(open(filename,"rt",encoding="utf-8")):
            sentences.append(["<START>"]+[t[FORM] for t in sent]+["<END>"])
            labels.append(["<START>"]+[t[UPOS] for t in sent]+["<END>"])
            if count_words:
                for word in sentences[-1]:
                    self.word_counts[word]=self.word_counts.get(word,0)+1
            if len(sentences)>=max_sent:
                break
        return sentences,labels


    def mask_rare_words(self, sentences, freq=5, mask_term="__UNK__",verbose=False):
        if self.word_counts==None:
            print("Cannot mask because we don't have word_counts.")
            return
        masked_sentences=[]
        for sent in sentences:
            new_sent=[]
            for word in sent:
                if self.word_counts.get(word,0)<freq:
                    if verbose:
                        print("Masking word",word,"with frequence",self.word_counts.get(word,0))
                    new_sent.append(mask_term)
                else:
                    new_sent.append(word)
            masked_sentences.append(sent)
        return masked_sentences


class TorchData(object):

    def __init__(self):

        # create text vectorizers
        self.word_vectorizer=text_vectorizer(with_unknown="__UNK__")
        self.char_vectorizer=text_vectorizer(with_unknown="__UNK__")
        self.label_vectorizer=text_vectorizer(with_unknown=None)

    def shuffle_two(self,one,two):

        three=[(a,b) for a,b in zip(one,two)]
        shuffle(three)
        one=[a for a,b in three]
        two=[b for a,b in three]
        return one, two

    def minibatched_3dim(self,data,batch_size):
        seq_count,word_count,char_count=data.size()
        seq_count_mbatch_aligned=(seq_count//batch_size)*batch_size
        data_batched=data[:seq_count_mbatch_aligned].transpose(0,1).contiguous().view(word_count,seq_count//batch_size,-1,char_count)
        return data_batched

    def prepare_torch_data(self, sentences, labels, batch_size, seq_words, seq_chars, train=True, shuffle=False):

        if shuffle:
            sentences,labels=self.shuffle_two(sentences,labels)
        # vectorize
        data_words=self.word_vectorizer(sentences,train=train)
        data_chars=self.char_vectorizer(sentences,string_as_sequence=True,train=train)
        data_labels=self.label_vectorizer(labels) # TODO: what if dev/test have labels not present in train?

        # torch tensor
        tdata,_=to_torch_long_tensor(data_words,[seq_words])
        cdata,_=to_torch_long_tensor(data_chars,[seq_words, seq_chars])
        ldata,_=to_torch_long_tensor(data_labels,[seq_words])
            
        # minibatch
        batches_word=torch_minibatched_2dim(tdata,batch_size)
        batches_char=self.minibatched_3dim(cdata,batch_size)
        batches_label=torch_minibatched_2dim(ldata,batch_size)

        return batches_word, batches_char, batches_label


