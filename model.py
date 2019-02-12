import tensorflow as tf
from ModelLayers import Embedding,ConvLayer
from PredictMethod import Classification,Jugde,LossCount,Optimizer

class RC_model():
    def __init__(self,max_length,embedding_size,pos_tot,pos_embedding_size,word_vec_mat,filter_size,filter_num,
                 hidden_size,class_num,learning_rate):
        self.pos_tot=pos_tot
        self.pos_embedding_size=pos_embedding_size
        self.embedding_size=embedding_size
        self.max_length=100
        self.word_vec_mat=word_vec_mat

        self.filter_size=filter_size
        self.filter_num=filter_num

        self.hidden_size=hidden_size
        self.learning_rate=learning_rate

        self.class_num=class_num

    def _placehold_init(self):
        self.sentence1_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_sentence1')
        self.sentence2_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_sentence2')
        self.pos1_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_pos1')
        self.pos2_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_pos2')
        self.pos1_cross_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_cross_pos1')
        self.pos2_cross_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_cross_pos2')
        self.label_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,self.class_num],name='label_holder')

    def _embedding(self):
        self.sentence1_embedding=Embedding.word_embedding(self.sentence1_placeholder,self.word_vec_mat,self.embedding_size)
        self.sentence2_embedding=Embedding.word_embedding(self.sentence2_placeholder,self.word_vec_mat,self.embedding_size)

        self.pos1=Embedding.pos_embedding(self.pos1_placeholder,self.pos_tot,self.pos_embedding_size)
        self.pos2=Embedding.pos_embedding(self.pos2_placeholder,self.pos_tot,self.pos_embedding_size)

        self.pos1_cross=Embedding.pos_embedding(self.pos1_cross_placeholder,self.pos_tot,self.pos_embedding_size)
        self.pos2_cross=Embedding.pos_embedding(self.pos2_cross_placeholder,self.pos_tot,self.pos_embedding_size)

        self.sentence1_embedding=tf.concat([self.sentence1_embedding,self.pos1,self.pos1_cross],axis=-1)
        self.sentence2_embedding=tf.concat([self.sentence2_embedding,self.pos2,self.pos2_cross],axis=-1)


    def _encode(self):
        self.text_sentence=tf.concat([self.sentence1_embedding,self.sentence2_embedding],axis=1)

    def _method(self):
        self.text_sentence=ConvLayer.text_conv(self.text_sentence,self.filter_size,self.filter_num)

    def _result(self):
        self.classification_result=Classification.logits_classification\
            (self.text_sentence,self.hidden_size,self.class_num,mode='train')
        self.loss=LossCount.loss_count(self.classification_result,self.label_placeholder)

    def _optimizer(self):
        self.train_op=Optimizer.optimizer(self.learning_rate,self.loss)

    def build_model(self):
        self._placehold_init()
        self._embedding()
        self._encode()
        self._method()
        self._result()
        self._optimizer()




