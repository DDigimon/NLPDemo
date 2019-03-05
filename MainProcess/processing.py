from DataSet import dataset

import yaml

def processing():
    with open('./config.yaml',encoding='utf-8') as file_config:
        config=yaml.load(file_config)

    data=dataset(config)
    data.init_data()
    data.read_wordvec()
    data.data_reader(mode='train')
    data.data_reader(mode='valid')
    data.data_reader(mode='test')
    data.data_reader(mode='local_test')
    data.save_data()
# print(data.label_count)