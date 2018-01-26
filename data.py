import sys
import torch
import numpy as np

from text_classifier_torch.t2i import T2I as text_vectorizer
from text_classifier_torch.t2i import to_torch_long_tensor,torch_minibatched_2dim

WARNING='\033[91m'
END_WARNING='\033[0m'

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
            print(WARNING, "Warning! Cannot mask because we don't have word_counts. Use count_words=True when reading data.", END_WARNING, file=sys.stderr)
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
        self.pretrained_embedding_model=None
        self.vidx=(0,0)
        self.embedding_size=None


    def init_vocab_from_pretrained(self,embedding_file, vocab_size=500000):
        try:
            import lwvlib
        except:
            print(WARNING, "Warning! Could not import lwvlib, pretrained embeddings cannot be loaded.", END_WARNING, file=sys.stderr)
            sys.exit()
        self.pretrained_embedding_model=lwvlib.load(embedding_file, max_rank_mem=500000, max_rank=500000) # TODO
        pre=len(self.word_vectorizer.idict)
        # init the vectorizer dictionary based on words in our model
        _=self.word_vectorizer(self.pretrained_embedding_model.words[:vocab_size],train=True)
        self.vidx=(pre,vocab_size)
        self.embedding_size=self.pretrained_embedding_model.vectors.shape[1]
        print(vocab_size, "word read from embedding model, dimensionality:", self.embedding_size)

     
    def load_pretrained_embeddings(self, embedding_file, model, model_word_embeddings):
        # vectorizer has token to index dictionary
        if self.pretrained_embedding_model is None:
            print(WARNING, "Cannot load pretrained embeddings, did you call init_vocab_from_pretrained()?", END_WARNING, file=sys.stderr)
            sys.exit()

        new_weights = np.zeros((len(self.word_vectorizer.idict),self.embedding_size))
        print("Loading pretrained word embeddings from", embedding_file)

        embeddings=self.pretrained_embedding_model.vectors[:self.vidx[1], :]
        new_weights[self.vidx[0]:self.vidx[0]+self.vidx[1],:]=embeddings # replace part of the matrix, rest is still zeros
        model_word_embeddings.weight.data.copy_(torch.from_numpy(new_weights))

        self.pretrained_embedding_model=None # no need for this anymore, free the memory

        return



    def shuffle_two(self,one,two):

        three=[(a,b) for a,b in zip(one,two)]
        np.random.shuffle(three)
        one=[a for a,b in three]
        two=[b for a,b in three]
        return one, two

    def minibatched_3dim(self,data,batch_size):
        seq_count,word_count,char_count=data.size()
        seq_count_mbatch_aligned=(seq_count//batch_size)*batch_size
        data_batched=data[:seq_count_mbatch_aligned].transpose(0,1).contiguous().view(word_count,seq_count//batch_size,-1,char_count)
        return data_batched

    def prepare_torch_data(self, sentences, labels, batch_size, seq_words, seq_chars, train=True, shuffle=False, sort_batch=False):

        if shuffle:
            sentences,labels=self.shuffle_two(sentences,labels)
        # vectorize
        data_words=self.word_vectorizer(sentences,train=train)
        data_chars=self.char_vectorizer(sentences,string_as_sequence=True,train=train)
        data_labels=self.label_vectorizer(labels) # TODO: what if dev/test have labels not present in train?

        # torch tensor
        tdata,tlens=to_torch_long_tensor(data_words,[seq_words])
        cdata,_=to_torch_long_tensor(data_chars,[seq_words, seq_chars])
        ldata,_=to_torch_long_tensor(data_labels,[seq_words])
        
        if sort_batch: # sort data inside batches in descending order, TODO: do this in pytorch not numpy, should not sort the whole treebank, just inside minibatches?

            sorting_indices=tlens.numpy().argsort()[::-1] # descending order
            
            sorted_tlens_np=tlens.numpy()[sorting_indices]
            sorted_tdata_np=tdata.numpy()[sorting_indices]
            sorted_cdata_np=cdata.numpy()[sorting_indices]
            sorted_ldata_np=ldata.numpy()[sorting_indices]

            tlens=torch.from_numpy(sorted_tlens_np)
            tdata=torch.from_numpy(sorted_tdata_np)
            cdata=torch.from_numpy(sorted_cdata_np)
            ldata=torch.from_numpy(sorted_ldata_np)

            unsorting_indices=sorting_indices.argsort()
#            original_back=sorted_tlens_np[unsorting_indices]
#            print("unsorted data:",original_back)
#            print("original==original.sorted().unsorted():",tlens_np==original_back)
        else:
            unsorting_indices=None
            
        # minibatch
        batches_word=torch_minibatched_2dim(tdata,batch_size)
        batches_char=self.minibatched_3dim(cdata,batch_size)
        batches_label=torch_minibatched_2dim(ldata,batch_size)

        return batches_word, batches_char, batches_label, tlens, sorting_indices, unsorting_indices





