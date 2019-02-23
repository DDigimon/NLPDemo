import os
# from load_data import DataProcess
import pickle
import numpy as np
from collections import defaultdict
# from utils import create_dictionary
symbol_list=['\n','\r\n','\t',' ']
break_list=['。','；']
max_length=100
def _get_relation(ann_path):
    re_list=[]
    with open(ann_path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split('\n')[0].split('\t')
            if len(line) <= 1: continue
            if line[0][0] == 'R':
                line[1]=line[1].split(' ')
                line[1][2]=line[1][2].split(':')
                line[1][1]=line[1][1].split(':')
                re_list.append((line[1][1][1],line[1][2][1]))
    return re_list

def _segement_txt(txt_path):
    dic={}
    with open(txt_path,encoding='utf-8') as f:
        file=f.read()
        string=''
        for id,c in enumerate(file):
            if c=='\n':
                c=' '
            if c in break_list:
                dic[id]=string
                string=''
            string+=c
        dic[id+1]=string
    return dic,file

def _get_entity(ann_path):
    entity_dic={}
    with open(ann_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split('\t')
            if len(line)<=1:continue
            if line[0][0]=='T':
                # 仅获取实体
                line[1]=line[1].split(' ')
                entity_dic[line[0]]={}
                entity_dic[line[0]]['type']=line[1][0]
                entity_dic[line[0]]['name']=line[2]
                entity_dic[line[0]]['start']=int(line[1][1])
                # print(line)
    # print(entity_dic)
    return entity_dic

def _find_relation(entity_dic,limit_lens):
    result_set = []
    count = 0
    for key in entity_dic.keys():
        for key2 in entity_dic.keys():
            if abs(entity_dic[key]['start']-entity_dic[key2]['start'])<limit_lens:
                if key!=key2:
                    if entity_dic[key2]['type']=='Disease' or entity_dic[key2]['type']=='Drug':
                        if (key,key2) not in result_set:
                            result_set.append((key,key2))
                            count+=1
    return result_set

def _GetString(document_dic,entity_dic,result_set,file,re_list=None,ngram=100):
    def _remove_symbol_for_ngram(string,id,direction=1):
        # 去除分隔符干扰
        while 1:
            flag = False
            for c in string:
                if c in symbol_list:
                    flag=True
            if flag==False:break
            count = 0
            for c in string:
                if c in symbol_list:
                    count += 1
            for c in symbol_list:
                string=string.replace(c,'')
            if direction==1:
                tmp=file[id:id+count+ngram*3]
                for c in symbol_list:
                    tmp=tmp.replace(c,'')
                # print(tmp)
                string=tmp[:ngram]
            else:
                tmp = file[id-count-ngram*3:id+1]
                for c in symbol_list:
                    tmp = tmp.replace(c, '')
                string=tmp[:ngram]
        return string
    def _remove_symbol(string):
        for c in symbol_list:
            string=string.replace(c,'')
        return string
    def _remove_both_symbol(entity,string):
        new_string=''
        start=entity[0]
        length=entity[1]
        for id,c in enumerate(string):
            if c in symbol_list:
                if id<entity[0]:
                    start-=1
                elif id>=entity[0] and id<entity[0]+entity[1]:
                    length-=1
            else:
                new_string+=c
        while len(new_string)>max_length:
            mid=int(len(new_string)/2)
            if start<mid:
                new_string=new_string[:mid]
            else:
                new_string=new_string[mid:]
                start-=mid
        if start+length>len(new_string):
            length=len(new_string)-start

        return [start,length],new_string

    document_set=[]
    # 生成断句id列表
    key_set=[0]
    for key in document_dic.keys():
        key_set.append(key)
    key_set.append(key+1000)

    # print(result_set)
    # 遍历列表
    count=0
    for pair in result_set:
        # print(pair[0])
        if entity_dic[pair[0]]['name']==entity_dic[pair[1]]['name']:continue
        count+=1
        string_list=['','']
        string=''
        start_id = entity_dic[pair[0]]['start']
        start_name = entity_dic[pair[0]]['name']
        end_id=entity_dic[pair[1]]['start']
        end_name=entity_dic[pair[1]]['name']
        start_entity=(0,0)
        end_entity=(0,0)
        for i in range(len(key_set)-1):
            if start_id<key_set[i] and start_id>=key_set[i-1]:
                start_entity=(start_id-key_set[i-1],len(start_name))
                start_entity,doc_string=_remove_both_symbol(start_entity, document_dic[key_set[i]])
                string_list.append(doc_string)
                # print(start_entity,len(doc_string))
                # neibor_start=(_remove_symbol_for_ngram(file[start_id-ngram:start_id],start_id,0),_remove_symbol(start_name),
                #               _remove_symbol_for_ngram(file[start_id+len(start_name):start_id+len(start_name)+ngram],
                #                              start_id+len(start_name),1))
                # print(start_entity)

        for i in range(len(key_set)-1):
            if end_id<key_set[i] and end_id>=key_set[i-1]:
                end_entity=(end_id-key_set[i-1],len(end_name))
                end_entity,doc_string=_remove_both_symbol(end_entity,document_dic[key_set[i]])
                string_list.append(doc_string)
                # neibor_end=(_remove_symbol_for_ngram(file[end_id-ngram:end_id],end_id,0),_remove_symbol(end_name),
                #             _remove_symbol_for_ngram(file[end_id+len(end_name):end_id+len(end_name)+ngram],
                #                            end_id+len(end_name),1))

        dis=entity_dic[pair[0]]['start']-entity_dic[pair[1]]['start']
        if dis==0:
            string=string_list[0]+' '+string_list[0]
        else:
            string=string_list[0]+' '+string_list[1]
        if re_list!=None:
            if pair in re_list:
                flag=1
            else:
                flag=0
        else:
            flag=0
        string=str(start_entity[0])+' '+str(start_entity[1])+' '+str(start_id)+'\t'\
               +str(end_entity[0])+' '+str(end_entity[1])+' '+str(end_id)+'\t'\
               +str(flag)+'\t'+string+'\n'
        document_set.append(string)
        # print(string)
    # print(document_set)
    return document_set

def GenerateTrainSet(ori_path,aim_merge_path):
    root=ori_path
    file_list=[]
    with open(aim_merge_path,'w',encoding='utf-8') as fin:
        for files in os.listdir(root):
            files=files.split('.')[0]
            if files not in file_list:
                file_list.append(files)
                document_dic, file = _segement_txt(root+files+'.txt')
                entity_dic = _get_entity(root+files+'.ann')
                result_set = _find_relation(entity_dic, 200)
                re_list = _get_relation(root+files+'.ann')
                documents=_GetString(document_dic,entity_dic,result_set,file,re_list)
                for i in documents:
                    fin.write(i)

def GenerateTestSet(ori_path,aim_merge_path):
    root = ori_path
    file_list = []
    with open(aim_merge_path, 'w', encoding='utf-8') as fin:
        for files in os.listdir(root):
            files = files.split('.')[0]
            if files not in file_list:
                file_list.append(files)
                document_dic, file = _segement_txt(root + files + '.txt')
                entity_dic = _get_entity(root + files + '.ann')
                result_set = _find_relation(entity_dic, 200)
                # re_list = _get_relation(root + files + '.ann')
                documents = _GetString(document_dic, entity_dic, result_set, file)
                for i in documents:
                    fin.write(i)

