import sys
import torch
from numpy.random import shuffle

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

    def expand_vocabulary_from_pretrained(self, args, expand=100000):

        if len(args.pretrained_word_embeddings)==0:
            return None, len(self.word_vectorizer.idict)

        else: # we will load pretrained word embeddings if possible, vocabulary will be (current | words from vector model)
            try:
                import lwvlib
            except:
                print(WARNING, "Warning! Could not import lwvlib, pretrained embeddings cannot be loaded.", END_WARNING, file=sys.stderr)
                return None, len(self.word_vectorizer.idict)
            self.pretrained_embedding_model=lwvlib.load(args.pretrained_word_embeddings, max_rank_mem=500000, max_rank=500000) # TODO
            # expand the vectorizer dictionary based on words in our model
            _=self.word_vectorizer(self.pretrained_embedding_model.words[:expand],train=True)
            return self.pretrained_embedding_model.vectors.shape[1], len(self.word_vectorizer.idict)

    def load_pretrained_embeddings(self, embedding_file, vectorizer, model, model_word_embeddings):
        # vectorizer has token to index dictionary
        try:
            import lwvlib
        except:
            print(WARNING, "Warning! Could not import lwvlib, pretrained embeddings not loaded.", END_WARNING, file=sys.stderr)
            return

        # Initialize with pretrained embeddings
        new_weights = model_word_embeddings.weight.data # copy initialization for unknowns
        print("Loading pretrained word embeddings from", embedding_file)
        pretrained = {}
        emb_invalid = 0
        if not self.pretrained_embedding_model:
            print(WARNING, "Warning! Most likely will load pretrained weights only for words present in the training data. Try calling calculate_vocabulary_size() before initializing the nn model.", END_WARNING, file=sys.stderr)
            self.pretrained_embedding_model=lwvlib.load(embedding_file, max_rank_mem=500000, max_rank=500000) # TODO

#        if new_weights.size(1)!=self.pretrained_embedding_model.vectors.shape[1]:
#            print(WARNING, "Warning! Dimensionality mismatch,", new_weights.size(1), "vs", self.pretrained_embedding_model.vectors.shape[1]  ,", pretrained embeddings cannot be loaded.", END_WARNING, file=sys.stderr)
#            return

        found=0
        not_found=0
        # copy vectors TODO: very inefficient!
        for word,idx in vectorizer.idict.items():
            if word in self.pretrained_embedding_model.words:
                new_weights[idx] = torch.from_numpy(self.pretrained_embedding_model.vectors[self.pretrained_embedding_model.get(word)])
                found+=1
            elif word.lower() in self.pretrained_embedding_model.words:
                new_weights[idx] = torch.from_numpy(self.pretrained_embedding_model.vectors[self.pretrained_embedding_model.get(word.lower())])
                found+=1
            else:
                not_found+=1
        # replace weights in model
        model_word_embeddings.weight = torch.nn.Parameter(new_weights)

        print("Loaded", found, "pretrained embeddings from", embedding_file)
        print("Embeddings for", not_found, "words not found.")

        self.pretrained_embedding_model=None # no need for this anymore, free the memory

        return


