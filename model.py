import tensorflow as tf
import numpy as np
from ModelLayers import Embedding,ConvLayer
from PredictMethod import Classification,Jugde,LossCount,Optimizer


class RC_model():
    def __init__(self,config,mode='train'):
        self.pos_tot=config['param']['max_pos_id']
        self.pos_embedding_size=int(config['param']['pos_embedding_size'])
        self.embedding_size=int(config['param']['embedding_size'])
        self.max_length=int(config['param']['max_length'])
        self.word_vec_mat=np.load(config['file_path']['word_vec_np'])

        self.filter_size=int(config['param']['filter_size'])
        self.filter_num=int(config['param']['filter_num'])

        self.hidden_size=int(config['param']['hidden_size'])
        self.learning_rate=int(config['param']['learning_rate'])

        self.class_num=int(config['param']['class_num'])
        self.dropout_rate=int(config['param']['dropout_rate'])

        self.mode=mode

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
        if self.mode=='train':
            self.text_sentence=tf.nn.dropout(self.text_sentence,keep_prob=1-self.dropout_rate,name='encode_dropout')

    def _method(self):
        self.text_sentence=ConvLayer.text_conv(self.text_sentence,self.filter_size,self.filter_num,
                                               mode='train',dropout_rate=self.dropout_rate)
        if self.mode=='train':
            self.text_sentence=tf.nn.dropout(self.text_sentence,keep_prob=1-self.dropout_rate,name='method_dropout')

    def _result(self):
        self.classification_result,self.pred=Classification.logits_classification\
            (self.text_sentence,self.hidden_size,self.class_num,mode=self.mode)
        self.loss=LossCount.loss_count(self.classification_result,self.label_placeholder)

    def _optimizer(self):
        self.train_op=Optimizer.optimizer(self.learning_rate,self.loss)

    def _judge(self):
        self.acc=Jugde.acc_count(self.pred,self.label_placeholder)

    def build_model(self):
        self._placehold_init()
        self._embedding()
        self._encode()
        self._method()
        self._result()
        self._optimizer()
        if self.mode=='train':
            self._judge()




