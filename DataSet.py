import os
import random
import numpy as np
import init
import yaml
import pickle
class dataset():
    def __init__(self,max_length):

        self.train_file='./data/train.txt'
        self.test_file='./data/test.txt'
        self.test_test_file='./data/test_test.txt'
        self.valid_file='./data/valid.txt'
        self.train_ori_file='./ori_data/train/'
        self.test_ori_file='./ori_data/tmp_test/'
        self.valid_ori_file='./ori_data/valid/'
        self.wordvec_file='./data/char.txt'

        self.max_id=10000
        self.embedding_size=0
        self.max_length=max_length
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
                # id_list.append(id-entity_pos)
                id_list.append(1)
            elif id >=entity_pos:
                if id<entity_pos+entity_length:
                    id_list.append(0)
                else:
                    # id_list.append(id-entity_pos-entity_length)
                    id_list.append(2)
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
            # start_id_list.append(id+dis)
            start_id_list.append(1)
            id=id
        if id<self.max_length:
            for _ in range(self.max_length-id-1):
                start_id_list.append(self.max_id)

        for id,c in enumerate(start_sentence):
            if id>=self.max_length:break
            # end_id_list.append(id-dis)
            end_id_list.append(2)
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
        # for files in os.listdir(self.valid_file):
        #     print(files)
        init.GenerateTrainSet(self.test_ori_file,self.test_file)
        init.GenerateTestSet(self.test_ori_file,self.test_test_file)
    # def match_num(self,string):
    #
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

        with open('./data/pickle/vec.pkl','wb') as f:
            pickle.dump(self.word_set,f)

        np.save('./data/vec',self.id_set)

    def read_train_data(self):
        with open(self.test_file,encoding='utf-8') as f:
            for line in f.readlines():
                line=line.split('\n')[0].split('\t')
                start_entity=line[0].split(' ')
                end_entity=line[1].split(' ')
                flag=line[2]
                start_sentence=line[3].split(' ')[0]
                end_sentence=line[3].split(' ')[1]
                if line[2] not in self.label_count:
                    self.label_count[line[2]]={}
                    self.label_count[line[2]]['id']=len(self.label_count)-1
                    self.label_count[line[2]]['value']=0
                self.label_count[line[2]]['value']+=1
                self.train_set[self.train_num]={}
                self.train_set[self.train_num]['start_id']=int(start_entity[0])
                self.train_set[self.train_num]['start_length']=int(start_entity[1])
                self.train_set[self.train_num]['start_sentence']=self._word2id(start_sentence)
                self.train_set[self.train_num]['end_id']=int(end_entity[0])
                self.train_set[self.train_num]['end_length']=int(end_entity[1])
                self.train_set[self.train_num]['end_sentence']=self._word2id(end_sentence)
                self.train_set[self.train_num]['start_sentence_self_id']=self._pos_id(start_sentence,
                                                                                   int(start_entity[0]),
                                                                                   int(start_entity[1]))
                # print(self.train_set[self.train_num]['start_sentence_self_id'])
                self.train_set[self.train_num]['end_sentence_self_id'] = self._pos_id(end_sentence,
                                                                                     int(end_entity[0]),
                                                                                     int(end_entity[1]))
                self.train_set[self.train_num]['start_sentence_cross_id'],\
                self.train_set[self.train_num]['end_sentence_cross_id']=\
                    self._cross_id(start_sentence,end_sentence,
                                   int(start_entity[2]),int(end_entity[2]),
                                   int(start_entity[0]),int(start_entity[0]))

                self.train_set[self.train_num]['start_sentence_real_length']=len(start_sentence)
                self.train_set[self.train_num]['end_sentence_real_length']=len(end_sentence)

                self.train_set[self.train_num]['flag']=self._flag2id(flag)
                self.train_num += 1

        with open('./data/pickle/train.pkl','wb') as f:
            pickle.dump(self.train_set,f)

        with open('./data/pickle/label.pkl','wb') as f:
            pickle.dump(self.label_count,f)

    def read_valid_data(self):
        with open(self.test_file,encoding='utf-8') as f:
            for line in f.readlines():
                line=line.split('\n')[0].split('\t')
                start_entity = line[0].split(' ')
                end_entity = line[1].split(' ')
                flag = line[2]
                start_sentence = line[3].split(' ')[0]
                end_sentence = line[3].split(' ')[1]

                self.valid_set[self.valid_num]={}
                self.valid_set[self.valid_num]['start_id']=int(start_entity[0])
                self.valid_set[self.valid_num]['start_length']=int(start_entity[1])
                self.valid_set[self.valid_num]['start_sentence']=self._word2id(start_sentence)
                self.valid_set[self.valid_num]['end_id']=int(end_entity[0])
                self.valid_set[self.valid_num]['end_length']=int(end_entity[1])
                self.valid_set[self.valid_num]['end_sentence']=self._word2id(end_sentence)
                self.valid_set[self.valid_num]['start_sentence_self_id'] = self._pos_id(start_sentence,
                                                                                        int(start_entity[0]),
                                                                                        int(start_entity[1]))
                # print(self.train_set[self.train_num]['start_sentence_self_id'])
                self.valid_set[self.valid_num]['end_sentence_self_id'] = self._pos_id(end_sentence,
                                                                                      int(end_entity[0]),
                                                                                      int(end_entity[1]))
                self.valid_set[self.valid_num]['start_sentence_cross_id'], \
                self.valid_set[self.valid_num]['end_sentence_cross_id'] = \
                    self._cross_id(start_sentence, end_sentence,
                                   int(start_entity[2]), int(end_entity[2]),
                                   int(start_entity[0]), int(start_entity[0]))

                self.valid_set[self.valid_num]['start_sentence_real_length'] = len(start_sentence)
                self.valid_set[self.valid_num]['end_sentence_real_length'] = len(end_sentence)
                self.valid_set[self.valid_num]['flag']=self._flag2id(flag)
                self.valid_num += 1

        with open('./data/pickle/valid.pkl', 'wb') as f:
            pickle.dump(self.valid_set, f)

    def read_test_data(self):
        with open(self.test_file,encoding='utf-8') as f:
            for line in f.readlines():
                line=line.split('\n')[0].split('\t')
                start_entity = line[0].split(' ')
                end_entity = line[1].split(' ')
                start_sentence = line[3].split(' ')[0]
                end_sentence = line[3].split(' ')[1]

                self.test_set[self.test_num]={}
                self.test_set[self.test_num]['start_id']=int(start_entity[0])
                self.test_set[self.test_num]['start_length']=int(start_entity[1])
                self.test_set[self.test_num]['start_sentence']=self._word2id(start_sentence)
                self.test_set[self.test_num]['end_id']=int(end_entity[0])
                self.test_set[self.test_num]['end_length']=int(end_entity[1])
                self.test_set[self.test_num]['end_sentence']=self._word2id(end_sentence)
                self.test_set[self.test_num]['start_sentence_self_id'] = self._pos_id(start_sentence,
                                                                                      int(start_entity[0]),
                                                                                      int(start_entity[1]))
                # print(self.train_set[self.train_num]['start_sentence_self_id'])
                self.test_set[self.test_num]['end_sentence_self_id'] = self._pos_id(end_sentence,
                                                                                      int(end_entity[0]),
                                                                                      int(end_entity[1]))
                self.test_set[self.test_num]['start_sentence_cross_id'], \
                self.test_set[self.test_num]['end_sentence_cross_id'] = \
                    self._cross_id(start_sentence, end_sentence,
                                   int(start_entity[2]), int(end_entity[2]),
                                   int(start_entity[0]), int(start_entity[0]))

                self.test_set[self.test_num]['start_sentence_real_length'] = len(start_sentence)
                self.test_set[self.test_num]['end_sentence_real_length'] = len(end_sentence)
                self.test_num += 1
        with open('./data/pickle/test.pkl','wb') as f:
            pickle.dump(self.test_set,f)

    def save_data(self):
        with open('config.yaml',encoding='utf-8') as f:
            self.config=yaml.load(f)
        self.config['data_info']={}
        self.config['data_info']['train_data_num']=self.train_num
        self.config['data_info']['label']={}

        for key in self.label_count:
            self.config['data_info']['label'][key]=self.label_count[key]

        with open('config.yaml','w',encoding='utf-8') as f:
            yaml.dump(self.config,f)

    def load_data(self):
        with open('./data/pickle/train.pkl','rb') as f:
            self.train_set=pickle.load(f)
        with open('./data/pickle/valid.pkl','rb') as f:
            self.valid_set=pickle.load(f)
        with open('./data/pickle/test.pkl','rb') as f:
            self.test_set=pickle.load(f)
        with open('./data/pickle/label.pkl','rb') as f:
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
        self.is_break=False
        self.each_type_data={}
        self.each_type_data_num={}
        self.batch_id={}
        self.real_batch_num=0
        self.idx=0
        self.real_batch=0
        self.each_batch_flag_num = {}

        # for unbalance data
        if mode!='test':
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

        if mode!='test':
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
        else:
            for _ in range(self.real_batch_num):
                batch_list.append(self.idx)
                self.idx+=1
                if self.idx>=self.test_num:
                    self.idx=0
                    self.is_break=True


        random.shuffle(batch_list)

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