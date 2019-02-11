from model import RC_model
from DataSet import dataset
import numpy as np
from tqdm import tqdm
import tensorflow as tf


class train_method():
    def __init__(self, epoch, batch_size, max_patient, model, data,model_save_path):
        self.epoch=epoch
        self.model=model
        self.batch_size=batch_size
        self.data=data
        self.max_patient=max_patient
        self.model_save_path=model_save_path

        self.train_config()

    def feed_method(self,batch_data):
        feed_dic={self.model.sentence1_placeholder:batch_data['start_sentence'],
                  self.model.sentence2_placeholder:batch_data['end_sentence'],

                  self.model.pos1_placeholder:batch_data['start_sentence_self_id'],
                  self.model.pos2_placeholder:batch_data['end_sentence_self_id'],

                  self.model.pos1_cross_placeholder:batch_data['start_sentence_cross_id'],
                  self.model.pos2_cross_placeholder:batch_data['end_sentence_cross_id'],

                  self.model.label_placeholder:batch_data['flag']}
        return feed_dic

    def train_config(self):
        self.gpu_option=tf.GPUOptions(visible_device_list='0', allow_growth=True)
        self.sess=tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_option))

        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver()


    def train(self):
        for step in range(self.epoch):
            print('Epoch %d'%(step+1))

            train_loss=0.
            min_valid_loss=1000
            current_patient=0

            self.data.batch_data_init(self.batch_size,mode='train')
            real_batch_num=self.data.real_batch
            for _ in tqdm(range(real_batch_num)):
                batch_data=self.data.each_batch()
                # print(self.data.is_break)

                feed_dic=self.feed_method(batch_data)

                _,loss=self.sess.run([self.model.train_op,self.model.loss],feed_dict=feed_dic)
                train_loss+=loss

            if real_batch_num!=0:train_loss/=float(real_batch_num)
            valid_loss=self.evaluate()

            print('train loss:',train_loss,'valid loss',valid_loss)

            if valid_loss<min_valid_loss:
                min_valid_loss=valid_loss
                current_patient=0
                self.saver.save(self.sess,self.model_save_path)
            else:
                current_patient+=1


    def evaluate(self):
        eval_loss=0.
        self.data.batch_data_init(self.batch_size,mode='valid')
        real_batch_num=self.data.real_batch
        for _ in real_batch_num:
            batch_data=self.data.each_batch()

            feed_dic=self.feed_method(batch_data)
            loss=self.sess.run(self.model.loss,feed_dict=feed_dic)
            eval_loss+=loss

        if real_batch_num!=0:
            eval_loss/=float(real_batch_num)
        return eval_loss









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
max_patient=2
model_save_path='./data/models/'

model=RC_model(max_length=100,embedding_size=embedding_size,pos_tot=pos_tot,pos_embedding_size=pos_embedding,
               word_vec_mat=word_mat,filter_size=filter_size,filter_num=filter_num,hidden_size=hidden_size,
               class_num=class_num,learning_rate=learning_rate)
model.build_model()

data=dataset(max_length)

# data.load_data()
# data.load_init()
# data.read_valid_data()
data.load_data()
data.load_init()


print('read complite')
train=train_method(epoch=epoch, batch_size=batch_size, max_patient=max_patient, model=model, data=data,
                   model_save_path=model_save_path)

train.train()