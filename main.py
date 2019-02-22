import yaml
import argparse
from train_method import train_method
from test_method import test_method
from DataSet import dataset
from model import RC_model
import processing

def para_arg():
    paraser=argparse.ArgumentParser('NLPDemo')
    paraser.add_argument('--mode',type=str,default='local_test',help='mode choose')
    paraser.add_argument('--IsPro',type=bool,default=True,help='if need processing')

    return paraser

if __name__ == '__main__':
    arg = para_arg().parse_args()

    if arg.IsPro==False:
        processing.processing()

    with open('config.yaml',encoding='utf-8') as config_file:
        config=yaml.load(config_file)
    data=dataset(config)
    data.load_data(arg.mode)
    data.load_init()
    # print(data.label_count)

    if arg.mode=='train':
        model=RC_model(config,mode='train')
    else:
        model=RC_model(config,mode='test')

    model.build_model()

    if arg.mode=='train':
        train=train_method(data=data,model=model,config=config)
        train.train()
    elif arg.mode=='test':
        test=test_method(data=data,model=model,config=config)
        test.test(arg.mode)
    elif arg.mode=='local_test':
        test=test_method(data=data,model=model,config=config)
        test.test(arg.mode)
        test.judge_local()



