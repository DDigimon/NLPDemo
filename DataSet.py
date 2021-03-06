import random
import numpy as np
from Edible import init
import yaml
import pickle
class dataset():
    def __init__(self,config):

        self.config=config

        self.train_file=self.config['file_path']['train_path']
        self.test_file=self.config['file_path']['test_path']
        self.test_local_file=self.config['file_path']['test_local_path']
        self.valid_file=self.config['file_path']['valid_path']

        self.train_ori_file=self.config['file_path']['train_ori_path']
        self.test_ori_file=self.config['file_path']['test_ori_path']
        self.test_local_ori_file=self.config['file_path']['test_local_ori_path']
        self.valid_ori_file=self.config['file_path']['valid_ori_path']

        self.wordvec_file=self.config['file_path']['word_vec_path']

        self.train_ready_pkl=self.config['file_path']['train_ready_pkl']
        self.test_ready_pkl = self.config['file_path']['test_ready_pkl']
        self.test_local_ready_pkl=self.config['file_path']['test_local_ready_pkl']
        self.valid_ready_pkl = self.config['file_path']['valid_ready_pkl']
        self.word_vev_pkl=self.config['file_path']['word_vec_pkl']
        self.label_pkl=self.config['file_path']['label_pkl']

        self.max_id=self.config['param']['max_pos_id']
        self.mid_id=int(self.max_id/2)
        self.embedding_size=0
        self.max_length=config['param']['max_length']
        self.word_num=0
        self.word_set={}
        self.word_num=0

        self.label_count={}
        self.train_num=0
        self.train_set={}

        self.valid_num=0
        self.valid_set={}

        self.test_num=0
        self.test_set={}

    def _flag_list(self,id):
        flag_list=np.zeros(self.class_num)
        flag_list[id]=1
        return flag_list


    def _pos_id(self,sentence,entity_pos,entity_length):
        id_list=[]
        id=0
        for id,c in enumerate(sentence):
            if id>=self.max_length:break
            if id<entity_pos:
                id_list.append(id-entity_pos+self.mid_id)
            elif id >=entity_pos:
                if id<entity_pos+entity_length:
                    id_list.append(0)
                else:
                    id_list.append(id-entity_pos-entity_length+self.mid_id)
            id=id
        if id<self.max_length:
            for _ in range(self.max_length-id-1):
                id_list.append(self.max_id)

        return np.array(id_list)

    def _cross_id(self,start_sentence,end_sentence,real_pos1,real_pos2,pos1,pos2):
        start_id_list=[]
        end_id_list=[]
        dis=(real_pos1-pos1)-(real_pos2-pos2)
        id=0

        for id,c in enumerate(end_sentence):
            if id>=self.max_length:break
            start_id_list.append(id+dis+self.mid_id)
            id=id
        if id<self.max_length:
            for _ in range(self.max_length-id-1):
                start_id_list.append(self.max_id)

        for id,c in enumerate(start_sentence):
            if id>=self.max_length:break
            end_id_list.append(id-dis+self.mid_id)
            id=id

        if id<self.max_length:
            for _ in range(self.max_length-id-1):
                end_id_list.append(self.max_id)


        return np.array(start_id_list),np.array(end_id_list)

    def _word2id(self, string):
        new_string = []
        idx=0
        for idx,c in enumerate(string):
            if idx>=self.max_length:break
            if c in self.word_set:
                new_string.append(self.word_set[c])
            else:
                new_string.append(0)
            idx=idx
        if idx<self.max_length:
            for _ in range(self.max_length-idx-1):
                new_string.append(self.word_num)
        return np.array(new_string)


    def _flag2id(self, string):
        return int(self.label_count[string]['id'])

    def init_data(self):
        init.GenerateTrainSet(self.train_ori_file, self.train_file)
        init.GenerateTrainSet(self.valid_ori_file, self.valid_file)
        init.GenerateTrainSet(self.test_local_ori_file, self.test_local_file)
        init.GenerateTestSet(self.test_ori_file, self.test_file)


    def read_wordvec(self):
        with open(self.wordvec_file,encoding='utf-8') as f:
            for id,line in enumerate(f.readlines()):
                line=line.split('\n')[0]
                line=line.split(' ')
                if len(line)==2:
                    self.word_num=int(line[0])
                    self.embedding_size=int(line[1])

                    # 0 for unknow,the last for padding, random for [-0.25,0.25]
                    self.word_set['UK'] = 0
                    self.id_set = np.random.random(self.embedding_size) / 4
                else:
                    self.word_set[line[0]]=id
                    tmp_arr=[]
                    for i in range(1,self.embedding_size+1):
                        tmp_arr.append(float(line[i]))
                    self.id_set=np.row_stack((self.id_set,np.array(tmp_arr)))

        # padding
        self.word_num=len(self.word_set)
        self.word_set['PAD']=self.word_num
        self.id_set=np.row_stack((self.id_set,np.random.random(self.embedding_size)/4))
        # print(self.word_set['UK'])

        with open(self.config['file_path']['word_vec_pkl'],'wb') as f:
            pickle.dump(self.word_set,f)

        np.save(self.config['file_path']['word_vec_np'],self.id_set)

    def data_reader(self,mode='train'):
        data_set={}
        data_num=0
        file=''
        if mode=='train':
            file=self.train_file
        if mode=='valid':
            file=self.valid_file
        if mode=='test':
            file=self.test_file
        if mode=='local_test':
            file=self.test_local_file
        with open(file,encoding='utf-8') as f:
            for line in f.readlines():
                data_set[data_num] = {}

                line = line.split('\n')[0].split('\t')
                start_entity = line[0].split(' ')
                end_entity = line[1].split(' ')
                if mode!='test':
                    flag = line[2]
                    start_sentence = line[3].split(' ')[0]
                    end_sentence = line[3].split(' ')[1]
                    if mode=='train':
                        if line[2] not in self.label_count:
                            self.label_count[line[2]] = {}
                            self.label_count[line[2]]['id'] = len(self.label_count) - 1
                            self.label_count[line[2]]['value'] = 0
                        self.label_count[line[2]]['value'] += 1
                    data_set[data_num]['flag'] = self._flag2id(flag)
                elif mode=='test':
                    start_sentence = line[3].split(' ')[0]
                    end_sentence = line[3].split(' ')[1]


                data_set[data_num]['start_id']=int(start_entity[0])
                data_set[data_num]['start_length']=int(start_entity[1])
                data_set[data_num]['start_sentence']=self._word2id(start_sentence)
                data_set[data_num]['end_id']=int(end_entity[0])
                data_set[data_num]['end_length']=int(end_entity[1])
                data_set[data_num]['end_sentence']=self._word2id(end_sentence)
                data_set[data_num]['start_sentence_self_id']=self._pos_id(start_sentence,
                                                                                   int(start_entity[0]),
                                                                                   int(start_entity[1]))
                # print(self.train_set[self.train_num]['start_sentence_self_id'])
                data_set[data_num]['end_sentence_self_id'] = self._pos_id(end_sentence,
                                                                                     int(end_entity[0]),
                                                                                     int(end_entity[1]))
                data_set[data_num]['start_sentence_cross_id'],\
                data_set[data_num]['end_sentence_cross_id']=\
                    self._cross_id(start_sentence,end_sentence,
                                   int(start_entity[2]),int(end_entity[2]),
                                   int(start_entity[0]),int(start_entity[0]))

                data_set[data_num]['start_sentence_real_length']=len(start_sentence)
                data_set[data_num]['end_sentence_real_length']=len(end_sentence)

                data_num+= 1
        if mode=='train':
            self.train_set=data_set
            self.train_num=data_num
            with open(self.train_ready_pkl, 'wb') as f:
                pickle.dump(self.train_set, f)

            with open(self.label_pkl, 'wb') as f:
                pickle.dump(self.label_count, f)
        if mode=='valid':
            self.valid_set=data_set
            self.valid_num=data_num
            with open(self.valid_ready_pkl, 'wb') as f:
                pickle.dump(self.valid_set, f)

        if mode=='test':
            self.test_set=data_set
            self.test_num=data_num
            with open(self.test_ready_pkl, 'wb') as f:
                pickle.dump(self.test_set, f)

        if mode=='local_test':
            self.test_local_set = data_set
            self.test_local_num = data_num
            with open(self.test_local_ready_pkl, 'wb') as f:
                pickle.dump(self.test_local_set, f)

    def save_data(self):
        with open('config.yaml',encoding='utf-8') as f:
            self.config=yaml.load(f)
        self.config['param']['embedding_size'] = self.embedding_size
        self.config['param']['class_num']=len(self.label_count)


        # self.config['param']['batch_size']=32
        # self.config['param']['epoch']=20
        # self.config['param']['max_patient']=20
        # self.config['param']['max_length']=100
        #
        # self.config['param']['hidden_size']=128
        # self.config['param']['learning_rate']=0.01
        # self.config['param']['filter_size']=3
        # self.config['param']['filter_num']=5


        with open('config.yaml','w',encoding='utf-8') as f:
            yaml.dump(self.config,f)

    def load_data(self,mode):
        if mode=='train':
            with open(self.config['file_path']['train_ready_pkl'],'rb') as f:
                self.train_set=pickle.load(f)
            with open(self.config['file_path']['valid_ready_pkl'],'rb') as f:
                self.valid_set=pickle.load(f)
        if mode=='test':
            with open(self.config['file_path']['test_ready_pkl'],'rb') as f:
                self.test_set=pickle.load(f)
        if mode=='local_test':
            with open(self.config['file_path']['test_local_ready_pkl'],'rb') as f:
                self.test_local_set=pickle.load(f)

        with open(self.config['file_path']['label_ready_pkl'],'rb') as f:
            self.label_count=pickle.load(f)

    def load_init(self):
        self.class_num=len(self.label_count)
        self.train_num=len(self.train_set)
        self.valid_num=len(self.valid_set)
        self.test_num=len(self.test_set)


    def batch_data_init(self,batch_size,mode='train'):
        # print(self.train_set)
        if mode=='train':
            self.feed_data=self.train_set
        if mode=='valid':
            self.feed_data=self.valid_set
        if mode=='test':
            self.feed_data=self.test_set
        if mode=='local_test':
            self.feed_data=self.test_local_set
        self.is_break=False
        self.each_type_data={}
        self.each_type_data_num={}
        self.batch_id={}
        self.real_batch_num=0
        self.idx=0
        self.real_batch=0
        self.each_batch_flag_num = {}

        # for unbalance data
        if mode!='test' and mode!='local_test':
            for label_key in self.label_count:
                key=self._flag2id(label_key)
                self.each_batch_flag_num[key] = round(self.label_count[label_key]['value'] / self.train_num * batch_size)
                if self.each_batch_flag_num[key] == 0:
                    self.each_batch_flag_num[key] = 1

                self.real_batch_num+=self.each_batch_flag_num[key]
                self.batch_id[key]=0
                self.each_type_data[key]=[]
        else:
            # useless?
            self.real_batch_num=batch_size


        # shuffle
        if mode=='train':
            for key in self.train_set:
                self.each_type_data[self.train_set[key]['flag']].append(key)
            for key in self.each_type_data.keys():
                random.shuffle(self.each_type_data[key])
        if mode=='valid':
            for key in self.valid_set:
                self.each_type_data[self.valid_set[key]['flag']].append(key)
            for key in self.each_type_data.keys():
                random.shuffle(self.each_type_data[key])


        # count real batch num
        self.real_batch = int(len(self.feed_data) / self.real_batch_num)
        if len(self.feed_data) % self.real_batch_num != 0:
            self.real_batch += 1
        # print(self.each_type_data)



    def each_batch(self,mode='train'):
        batch_data={}
        batch_data['start_sentence']=[]
        batch_data['end_sentence']=[]
        batch_data['end_sentence_self_id']=[]
        batch_data['start_sentence_self_id']=[]
        batch_data['flag']=[]
        batch_data['start_sentence_cross_id']=[]
        batch_data['end_sentence_cross_id']=[]
        batch_list=[]

        if mode!='test' and mode!='local_test':
            for id,key in enumerate(self.each_batch_flag_num.keys()):
                get_list=[]

                if self.batch_id[key]+self.each_batch_flag_num[key]>=len(self.each_type_data[key]):
                    res_num=len(self.each_type_data[key])-self.batch_id[key]
                    get_list=random.sample(self.each_type_data[key],res_num)
                    self.each_batch_flag_num[key]-=res_num
                    self.batch_id[key]=0
                    self.is_break=True

                for idx in range(self.batch_id[key],self.batch_id[key]+self.each_batch_flag_num[key]):
                    get_list.append(self.each_type_data[key][idx])

                self.batch_id[key]+=self.each_batch_flag_num[key]
                batch_list.extend(get_list)
            random.shuffle(batch_list)
        else:
            for _ in range(self.real_batch_num):
                batch_list.append(self.idx)
                self.idx+=1
                if self.idx>=self.test_num:
                    self.idx=0
                    self.is_break=True

        # find max batch length
        max_length=0
        for id in batch_list:
            if self.feed_data[id]['start_sentence_real_length']>max_length:
                max_length=self.feed_data[id]['start_sentence_real_length']
            if self.feed_data[id]['end_sentence_real_length']>max_length:
                max_length=self.feed_data[id]['end_sentence_real_length']

        for id in batch_list:
            batch_data['start_sentence'].append(self.feed_data[id]['start_sentence'][:self.max_length])
            batch_data['end_sentence'].append(self.feed_data[id]['end_sentence'][:self.max_length])
            batch_data['start_sentence_self_id'].append(self.feed_data[id]['start_sentence_self_id'][:self.max_length])
            batch_data['end_sentence_self_id'].append(self.feed_data[id]['end_sentence_self_id'][:self.max_length])
            batch_data['start_sentence_cross_id'].append(self.feed_data[id]['start_sentence_cross_id'][:self.max_length])
            batch_data['end_sentence_cross_id'].append(self.feed_data[id]['end_sentence_cross_id'][:self.max_length])

            if mode!='test':
                batch_data['flag'].append(self._flag_list(self.feed_data[id]['flag']))

        for key in batch_data:
            batch_data[key]=np.array(batch_data[key])

        return batch_data

# data=dataset(100)
# # # # data.init_data()
# # # # data.read_wordvec()
# # data.read_train_data()
# # data.read_valid_data()
# # print(data.train_num,data.valid_num)
# data.load_data()
# data.load_init()
# # # print(data.train_set)
#
# # count=0
# for _ in range(1):
#     data.batch_data_init(32,mode='test')
#     # print(data.each_batch())
#     while True:
#         data.each_batch(mode='test')
#         # print(data.idx,data.test_num)
#         if data.is_break==True:
#             break
# # data.read_valid_data()
# # data.read_test_data()
# # data.save_data()