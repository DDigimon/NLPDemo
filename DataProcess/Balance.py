import random


def add_data(data_set,label_set):
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

