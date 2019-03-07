import random
import pickle

class batch_data():
    def __init__(self,config,mode='train'):
        if mode=='train':
            with open(config['file_path']['train_ready_pkl'],'rb') as f:
                self.train_set=pickle.load(f)
            with open(config['file_path']['valid_ready_pkl'],'rb') as f:
                self.valid_set=pickle.load(f)
        if mode=='test':
            with open(config['file_path']['test_ready_pkl'],'rb') as f:
                self.test_set=pickle.load(f)
        if mode=='test_local':
            with open(config['file_path']['test_local_ready_pkl'],'rb') as f:
                self.test_local_set=pickle.load(f)
        with open(config['file_path']['label_ready_pkl'],'rb') as f:
            self.label_count=pickle.load(f)

        self.train_num=len(self.train_set)

    def batch_data_init(self, batch_size, mode='train'):
        # print(self.train_set)
        if mode == 'train':
            self.feed_data = self.train_set
        if mode == 'valid':
            self.feed_data = self.valid_set
        if mode == 'test':
            self.feed_data = self.test_set
        if mode == 'local_test':
            self.feed_data = self.test_local_set
        self.is_break = False
        self.each_type_data = {}
        self.each_type_data_num = {}
        self.batch_id = {}
        self.real_batch_num = 0
        self.idx = 0
        self.real_batch = 0
        self.each_batch_flag_num = {}

        # for unbalance data
        if mode != 'test' and mode != 'local_test':
            for label_key in self.label_count:
                key = self._flag2id(label_key)
                self.each_batch_flag_num[key] = round(
                    self.label_count[label_key]['value'] / self.train_num * batch_size)
                if self.each_batch_flag_num[key] == 0:
                    self.each_batch_flag_num[key] = 1

                self.real_batch_num += self.each_batch_flag_num[key]
                self.batch_id[key] = 0
                self.each_type_data[key] = []
        else:
            # useless?
            self.real_batch_num = batch_size

        # shuffle
        if mode == 'train':
            for key in self.train_set:
                self.each_type_data[self.train_set[key]['flag']].append(key)
            for key in self.each_type_data.keys():
                random.shuffle(self.each_type_data[key])
        if mode == 'valid':
            for key in self.valid_set:
                self.each_type_data[self.valid_set[key]['flag']].append(key)
            for key in self.each_type_data.keys():
                random.shuffle(self.each_type_data[key])

        # count real batch num
        self.real_batch = int(len(self.feed_data) / self.real_batch_num)
        if len(self.feed_data) % self.real_batch_num != 0:
            self.real_batch += 1
        # print(self.each_type_data)

    def each_batch(self, mode='train'):
        batch_data = {}
        batch_data['start_sentence'] = []
        batch_data['end_sentence'] = []
        batch_data['end_sentence_self_id'] = []
        batch_data['start_sentence_self_id'] = []
        batch_data['flag'] = []
        batch_data['start_sentence_cross_id'] = []
        batch_data['end_sentence_cross_id'] = []
        batch_list = []

        if mode != 'test' and mode != 'local_test':
            for id, key in enumerate(self.each_batch_flag_num.keys()):
                get_list = []

                if self.batch_id[key] + self.each_batch_flag_num[key] >= len(self.each_type_data[key]):
                    res_num = len(self.each_type_data[key]) - self.batch_id[key]
                    get_list = random.sample(self.each_type_data[key], res_num)
                    self.each_batch_flag_num[key] -= res_num
                    self.batch_id[key] = 0
                    self.is_break = True

                for idx in range(self.batch_id[key], self.batch_id[key] + self.each_batch_flag_num[key]):
                    get_list.append(self.each_type_data[key][idx])

                self.batch_id[key] += self.each_batch_flag_num[key]
                batch_list.extend(get_list)
            random.shuffle(batch_list)
        else:
            for _ in range(self.real_batch_num):
                batch_list.append(self.idx)
                self.idx += 1
                if self.idx >= self.test_num:
                    self.idx = 0
                    self.is_break = True

        # find max batch length
        max_length = 0
        for id in batch_list:
            if self.feed_data[id]['start_sentence_real_length'] > max_length:
                max_length = self.feed_data[id]['start_sentence_real_length']
            if self.feed_data[id]['end_sentence_real_length'] > max_length:
                max_length = self.feed_data[id]['end_sentence_real_length']

        for id in batch_list:
            batch_data['start_sentence'].append(self.feed_data[id]['start_sentence'][:self.max_length])
            batch_data['end_sentence'].append(self.feed_data[id]['end_sentence'][:self.max_length])
            batch_data['start_sentence_self_id'].append(
                self.feed_data[id]['start_sentence_self_id'][:self.max_length])
            batch_data['end_sentence_self_id'].append(self.feed_data[id]['end_sentence_self_id'][:self.max_length])
            batch_data['start_sentence_cross_id'].append(
                self.feed_data[id]['start_sentence_cross_id'][:self.max_length])
            batch_data['end_sentence_cross_id'].append(
                self.feed_data[id]['end_sentence_cross_id'][:self.max_length])

            if mode != 'test':
                batch_data['flag'].append(self._flag_list(self.feed_data[id]['flag']))

        for key in batch_data:
            batch_data[key] = np.array(batch_data[key])

        return batch_data