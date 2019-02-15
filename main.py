import yaml
import argparse
from train_method import train_method
from DataSet import dataset
from model import RC_model

def para_arg():
    paraser=argparse.ArgumentParser('NLPDemo')
    paraser.add_argument('--mode',type=str,default='train',help='mode choose')

    return paraser

if __name__ == '__main__':
    with open('config.yaml',encoding='utf-8') as config_file:
        config=yaml.load(config_file)

    arg=para_arg().parse_args()

    data=dataset(config)
    data.load_data()
    data.load_init()

    if arg.mode=='train':
        model=RC_model(config,mode='train')
    else:
        model=RC_model(config,mode='test')

    model.build_model()

    train=train_method(data=data,model=model,config=config)
    train.train()



