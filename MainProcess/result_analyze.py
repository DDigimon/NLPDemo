import os

path='./ori_data/tmp_test/'

def file_ans(file_path):
    ans_list=[]
    with open(file_path,encoding='utf-8') as f:
        for line in f.readlines():
            if line[0]=='T' or len(line)<=1:
                continue
            ans_list.append(line.split('\n')[0].split('\t')[1])
    return ans_list


right_ans=0
wrong_ans=0
rest_ans=0
for file in os.listdir(path):
    if 'TEST' in file:
        ans_list=file_ans(path+file.split('TEST')[1])
        pred_list=file_ans(path+file)
        for ans in pred_list:
            if ans not in ans_list:
                wrong_ans+=1
            else:
                right_ans+=1

        for ans in ans_list:
            if ans not in ans_list:
                rest_ans+=1


print(right_ans,wrong_ans,rest_ans)
