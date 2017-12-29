

ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)
class POSdata(object):

    def __init__(self):
        pass
        

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


    def read_data(self,filename,max_sent):
        sentences=[]
        labels=[]
        for comm, sent in self.conllu_reader(open(filename,"rt",encoding="utf-8")):
            sentences.append([t[FORM] for t in sent])
            labels.append([t[UPOS] for t in sent])
            if len(sentences)>=max_sent:
                break
        return sentences,labels
