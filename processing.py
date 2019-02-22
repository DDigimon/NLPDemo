from DataSet import dataset

import yaml

with open('./config.yaml',encoding='utf-8') as file_config:
    config=yaml.load(file_config)

data=dataset(config)
data.init_data()
data.read_wordvec()
data.read_train_data()
data.read_valid_data()
data.read_test_data()
data.save_data()
# print(data.label_count)