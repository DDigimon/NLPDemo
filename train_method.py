from model import RC_model
from DataSet import dataset
import numpy as np
from tqdm import tqdm
import tensorflow as tf


class train_method():
    def __init__(self,epoch,batch_size,max_patient,model,train_data_set):
        self.epoch=epoch
        self.model=model
        self.batch_size=batch_size
        self.train_data=train_data_set
        self.max_patient=max_patient

        self.train_config()


    def train_config(self):
        self.gpu_option=tf.GPUOptions(visible_device_list='0', allow_growth=True)
        self.sess=tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_option))

        self.sess.run(tf.global_variables_initializer())


    def train(self):
        for step in range(self.epoch):
            print('Epoch %d'%(step+1))

            train_loss=0.
            min_dev_loss=1000
            current_patient=0

            self.train_data.batch_data_init(self.batch_size)
            for _ in tqdm(range(self.train_data.real_batch)):
                batch_data=self.train_data.each_batch(self.batch_size)
                feed_dic=self.feed_method(batch_data)

                _,loss=self.sess.run([self.model.train_op,self.model.loss],feed_dict=feed_dic)
                print(loss)


    def feed_method(self,batch_data):
        feed_dic={self.model.sentence1_placeholder:batch_data['start_sentence'],
                  self.model.sentence2_placeholder:batch_data['end_sentence'],

                  self.model.pos1_placeholder:batch_data['start_sentence_self_id'],
                  self.model.pos2_placeholder:batch_data['end_sentence_self_id'],

                  self.model.pos1_cross_placeholder:batch_data['start_sentence_cross_id'],
                  self.model.pos2_cross_placeholder:batch_data['end_sentence_cross_id'],

                  self.model.label_placeholder:batch_data['flag']}
        return feed_dic




max_length=100
embedding_size=300
pos_tot=10000
pos_embedding=50
word_mat=np.load('./data/vec.npy')
filter_size=3
filter_num=5
hidden_size=128
class_num=2
learning_rate=0.001
batch_size=32
epoch=1
max_patient=1

model=RC_model(max_length=100,embedding_size=embedding_size,pos_tot=pos_tot,pos_embedding_size=pos_embedding,
               word_vec_mat=word_mat,filter_size=filter_size,filter_num=filter_num,hidden_size=hidden_size,
               class_num=class_num,learning_rate=learning_rate)
model.build_model()

data=dataset(max_length)
data.init_data()
data.read_wordvec()
data.read_train_data()

train=train_method(epoch=epoch,batch_size=batch_size,max_patient=max_patient,model=model,train_data_set=data)

train.train()