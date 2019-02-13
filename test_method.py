from DataSet import dataset
from model import RC_model
from tqdm import tqdm
import tensorflow as tf
import numpy as np

class test_method():
    def __init__(self,data,model,model_save_path):
        self.max_length=100
        self.data=data
        self.model=model
        self.model_save_path=model_save_path

        self.session_config()

    def feed_method(self,batch_data):
        feed_dic={self.model.sentence1_placeholder:batch_data['start_sentence'],
                  self.model.sentence2_placeholder:batch_data['end_sentence'],

                  self.model.pos1_placeholder:batch_data['start_sentence_self_id'],
                  self.model.pos2_placeholder:batch_data['end_sentence_self_id'],

                  self.model.pos1_cross_placeholder:batch_data['start_sentence_cross_id'],
                  self.model.pos2_cross_placeholder:batch_data['end_sentence_cross_id']}
        return feed_dic

    def session_config(self):
        self.gpu_option = tf.GPUOptions(visible_device_list='0', allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_option))

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,self.model_save_path)

    def test(self):
        test_num=self.data.test_num
        pred_list=[]
        self.data.batch_data_init(1, mode='test')
        real_batch_num=self.data.real_batch
        for idx in tqdm(range(real_batch_num)):
            batch_data=self.data.each_batch(mode='test')
            feed_data=self.feed_method(batch_data)
            pred=self.sess.run(self.model.pred,feed_dict=feed_data)
            pred_list.extend(pred)
        print(pred_list)




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
               class_num=class_num,learning_rate=learning_rate,mode='test')
model.build_model()

data=dataset(max_length)

# data.load_data()
# data.load_init()
# data.read_valid_data()
data.load_data()
data.load_init()


print('read complite')
test=test_method(model=model, data=data,model_save_path=model_save_path)
test.test()