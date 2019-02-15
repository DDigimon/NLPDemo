import yaml
import argparse
from train_method import train_method
from DataSet import dataset
from model import RC_model


if __name__ == '__main__':
    mode='train'
    with open('config.yaml',encoding='utf-8') as config_file:
        config=yaml.load(config_file)


    data=dataset(config)
    data.load_data()
    data.load_init()

    if mode=='train':
        model=RC_model(config,mode='train')
        model.build_model()

    train=train_method(data=data,model=model,config=config)
    train.train()



