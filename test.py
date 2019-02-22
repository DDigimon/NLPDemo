# sentence='化,进而抑制病理性心肌肥厚;还可通过下调转化生长因子β1表达,从而抑制心肌内成纤维细胞胶原的合成[43]'
# start_id=9
# length=4
# real_start_id=7772
#
# end_sentence='。上述研究提示,二甲双胍除可减轻缺血-再灌注损伤外,还具有抑制心肌重构、延缓心力衰竭进展等心脏保护作用'
# end_pos=38
# end_length=4
# real_end_id=7918
#
# def __pos_id(sentence, entity_pos, entity_length):
#     id_list = []
#     for id, c in enumerate(sentence):
#         if id < entity_pos:
#             id_list.append(id - entity_pos)
#             print('a',id - entity_pos,c)
#         elif id >= entity_pos:
#             if id < entity_pos + entity_length:
#                 id_list.append(0)
#                 print('b',0,c)
#             else:
#                 id_list.append(id - entity_pos - entity_length)
#                 print('c',id - entity_pos - entity_length,c)
#     return id_list
#
# tmp_list=__pos_id(sentence,start_id,length)
#
# def cross_pos_id(sentence,end_id,start_id,real_pos1,real_pos2):
#     id_list=[]
#     entity_dis=(end_id-real_pos1)-(start_id-real_pos2)
#     print(end_id,real_pos1,start_id,real_pos2)
#     for id ,c in enumerate(sentence):
#         id_list.append(id-entity_dis)
#         print(id-entity_dis,c)
#
# cross_pos_id(end_sentence,end_pos,start_id,real_end_id,real_start_id)
#
# import numpy as np
# flag_list=np.zeros(4)
# flag_list[2]=1
# print(flag_list)

import yaml
import pickle
import random

with open('config.yaml', encoding='utf-8') as config_file:
    config = yaml.load(config_file)
with open(config['file_path']['valid_ready_pkl'], 'rb') as f:
    test_set = pickle.load(f)
with open(config['file_path']['label_ready_pkl'], 'rb') as f:
    label_count = pickle.load(f)

def add_data(data_set,label_set):
    res_label_set={}
    label_id_set={}
    data_label_set={}
    label_flag={}
    max_idx=0
    for key in label_set:
        label_id_set[label_set[key]['id']]=label_set[key]['value']
        label_flag[label_set[key]['id']]=False
    for idx,data in enumerate(data_set):
        flag=data_set[idx]['flag']
        if flag not in data_label_set:
            data_label_set[flag]=[]
        data_label_set[flag].append(idx)
        max_idx=idx
    max_num=label_id_set[max(label_id_set, key=label_id_set.get)]
    for key in label_id_set:
        if max_num<=label_id_set[key]:
            continue
        label_flag[key]=True
        res_num=max_num - label_id_set[key]
        tmp_list=data_label_set[key][:]
        while res_num>=(label_id_set[key]+len(tmp_list)):
            data_label_set[key].extend(tmp_list)
            label_id_set[key]+=len(tmp_list)
        data_label_set[key].extend(random.sample(data_label_set[key],(max_num-label_id_set[key])))

    for key in label_flag:
        if label_flag[key]==True:
            idx_list=[]
            if data_label_set[key] not in idx_list:
                idx_list.append(data_label_set[key])
            else:
                max_idx+=1
                data_set[max_idx]=data_set[data_label_set[key]]
    return data_set





add_data(test_set,label_count)