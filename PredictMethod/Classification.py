import tensorflow as tf
def logits_classification(input_layer,hidden_size,class_num,mode='test',var_scope=None):
    '''

    :param input_layer:
    :param hidden_size:
    :param class_num:
    :param mode: optional, test for default
    :param var_scope: optional
    :return: softmax result in test mode, logits result in train mode
    '''
    with tf.variable_scope(var_scope or 'logits_classification',reuse=tf.AUTO_REUSE):
        result_dense=tf.layers.dense(input_layer,hidden_size,name='d1')
        result_dense=tf.nn.relu(result_dense)
        logits=tf.layers.dense(result_dense,class_num,name='d2')
        pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        if mode=='test':
            return pred
        else:
            return logits
