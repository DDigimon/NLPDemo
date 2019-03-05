from gensim.models import word2vec
import numpy as np

def w2v(ori_path,result_path,embedding_size):
    sentence=word2vec.Text8Corpus(ori_path)
    model=word2vec.Word2Vec(sentences=sentence,size=embedding_size)
    word_dic={}
    word_dic['UK_'] = -1 + 2 * np.random.random(embedding_size)
    word_dic['BLK_'] = -1 + 2 * np.random.random(embedding_size)

    with open(ori_path,encoding='utf-8') as f:
        file=f.read()
        for c in file:
            if c!=' ' and c!='\n' and c!='\t' and c!='\r\n':
                if c not in word_dic:
                    if c in model:
                        word_dic[c]=model[c]
                    else:
                        word_dic[c]=-1+2*np.random.random(embedding_size)


    with open(result_path,'w',encoding='utf-8') as fin:
        fin.write(str(len(word_dic))+' '+str(embedding_size)+'\n')
        for c in word_dic:
            fin.write(c)
            for i in word_dic[c]:
                fin.write(' '+str(i))
            fin.write('\n')