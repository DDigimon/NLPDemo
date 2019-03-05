from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix


class test_method():
    def __init__(self,data,model,config):
        self.config=config
        self.data=data
        self.model=model
        self.model_save_path=config['file_path']['model_save_path']

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

    def test(self,mode):
        test_num=self.data.test_num
        self.pred_list=[]
        self.data.batch_data_init(1, mode=mode)
        real_batch_num=self.data.real_batch
        for idx in tqdm(range(real_batch_num)):
            batch_data=self.data.each_batch(mode=mode)
            feed_data=self.feed_method(batch_data)
            pred=self.sess.run(self.model.pred,feed_dict=feed_data)
            self.pred_list.extend(pred)

    def judge_local(self):
        y_true=[]
        for i in range(len(self.data.test_local_set)):
            y_true.append(self.data.test_local_set[i]['flag'])
        matrix=confusion_matrix(y_true,self.pred_list)
        print(matrix)





# max_length=100
# embedding_size=300
# pos_tot=10000
# pos_embedding=50
# word_mat=np.load('./data/vec.npy')
# filter_size=3
# filter_num=5
# hidden_size=128
# class_num=2
# learning_rate=0.001
# batch_size=32
# epoch=1
# max_patient=2
# model_save_path='./data/models/'
#
# model=RC_model(max_length=100,embedding_size=embedding_size,pos_tot=pos_tot,pos_embedding_size=pos_embedding,
#                word_vec_mat=word_mat,filter_size=filter_size,filter_num=filter_num,hidden_size=hidden_size,
#                class_num=class_num,learning_rate=learning_rate,mode='test')
# model.build_model()
#
# data=dataset(max_length)
#
# # data.load_data()
# # data.load_init()
# # data.read_valid_data()
# data.load_data()
# data.load_init()
#
#
# print('read complite')
# test=test_method(model=model, data=data,model_save_path=model_save_path)
# test.test()