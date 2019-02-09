import tensorflow as tf
def loss_count(logit_result,answer_result,var_scope=None):
    '''

    :param logit_result: logits from model layers
    :param answer_result: label from input
    :param var_scope: optional
    :return: loss value
    '''
    with tf.variable_scope(var_scope or 'cross_entropy',reuse=tf.AUTO_REUSE):
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logit_result,labels=answer_result)
        loss=tf.reduce_mean(cross_entropy)

        return loss
