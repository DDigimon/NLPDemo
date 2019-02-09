import tensorflow as tf

def _clip(optimizer,global_step,nil_grades_and_vars,clip_norm=10,var_scope=None):
    with tf.variable_scope(var_scope or None,reuse=tf.AUTO_REUSE):
        # global_step=tf.Variable(0,name='global_step',trainable=False)
        gradients,variables=zip(*nil_grades_and_vars)
        gradients,_=tf.clip_by_global_norm(gradients,clip_norm)
        train_op=optimizer.apply_gradients(zip(gradients,variables),name='train_op',global_step=global_step)

        return train_op


def optimizer(learning_rate,loss,clip=False,optimizer=tf.train.AdadeltaOptimizer):
    optimizer=optimizer(learning_rate)
    grads_and_vars=optimizer.compute_gradients(loss)
    nil_frads_and_vars=[]
    for g,v in grads_and_vars:
        nil_frads_and_vars.append((g,v))

    global_step=tf.Variable(0,name='global_step',trainable=False)

    if clip:
        train_op=_clip(optimizer,global_step,nil_frads_and_vars)
    else:
        train_op=optimizer.apply_gradients(nil_frads_and_vars,name='train_op',global_step=global_step)

    return train_op