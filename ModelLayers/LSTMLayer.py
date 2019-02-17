import tensorflow as tf
from tensorflow.contrib import rnn

def bi_lstm(embedding_input,hidden_size,sequence_length,var_scope=None,unit='lstm'):
    with tf.variable_scope(var_scope or 'text_conv',reuse=tf.AUTO_REUSE):
        embedding_input=tf.cast(embedding_input,dtype=tf.float32,name='change_float')
        if unit=='gru':
            fw_cell=rnn.GRUCell(hidden_size)
            bw_cell=rnn.GRUCell(hidden_size)
        else:
            fw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1, state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1, state_is_tuple=True)
        output,_=tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,embedding_input,scope='bi',
                                                 dtype=tf.float32,sequence_length=sequence_length)
        return output