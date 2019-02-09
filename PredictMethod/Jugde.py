import tensorflow as tf

def acc_count(pred,answer_result,var_scope=None):
    with tf.variable_scope(var_scope or 'acc',reuse=tf.AUTO_REUSE):
        correct_pred=tf.equal(tf.argmax(answer_result,axis=1),pred)
        acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        return acc