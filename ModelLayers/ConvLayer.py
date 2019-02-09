import tensorflow as tf

def text_conv(embedding_input,filter_size,filter_num,var_scope=None,pooling_method='max'):
    with tf.variable_scope(var_scope or 'text_conv',reuse=tf.AUTO_REUSE):
        embedding_input=tf.cast(embedding_input,dtype=tf.float32,name='change_float')
        conv=tf.layers.conv1d(embedding_input,filter_num,filter_size)
        if pooling_method=='max':
            output=tf.reduce_max(conv,reduction_indices=[1],name='global_pooling')
        elif pooling_method=='mean':
            output=tf.reduce_mean(conv,reduction_indices=[1],name='global_mean')
        else:
            output=tf.reduce_all(conv,reduction_indices=[1],name='global_all')
        return output