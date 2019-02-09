import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

def word_embedding(words,word_vec,embedding_size,var_scope=None,trainable=False,use_blank_id=False):
    '''

    :param words: sentence id list
    :param word_vec: pre_trained vector for words
    :param embedding_size: embedding size of word_vec
    :param var_scope: optional var_cape name
    :param trainable: embedding layer whether can be trained
    :param use_blank_id: the pre_trained vector map whether start with id 1
    :return: tf variable of sentence embedding
    '''
    with tf.variable_scope(var_scope or 'word_embedding',reuse=tf.AUTO_REUSE):
        word_embedding=tf.get_variable('word_embedding',initializer=word_vec,dtype=tf.float64,trainable=trainable)
        unk_embedding=tf.get_variable('unk_embedding',[1,embedding_size],dtype=tf.float64,
                                        initializer=tfc.layers.xavier_initializer())
        undefined_space=tf.constant(np.zeros((1,embedding_size),dtype=np.float32))
        if use_blank_id==False:
            word_embedding=tf.concat([word_embedding,unk_embedding],0)
        else:
            word_embedding=tf.concat([undefined_space,word_embedding,unk_embedding],0)
        output=tf.nn.embedding_lookup(word_embedding,words)
        print(output)

        return output

def pos_embedding(pos_list,pos_tot,pos_embedding_size,var_scope=None):
    '''

    :param pos_list: sentence pos id list
    :param pos_tot: the max pos id
    :param pos_embedding_size: position embedding size
    :param var_scope: optional var_cope name
    :return: tf variable of sentence id embedding
    '''
    with tf.variable_scope(var_scope or 'pos_embedding',reuse=tf.AUTO_REUSE):
        pos_embedding=tf.get_variable('pos_embedding',[pos_tot+1,pos_embedding_size],
                                      initializer=tfc.layers.xavier_initializer(),dtype=tf.float64)
        output=tf.nn.embedding_lookup(pos_embedding,pos_list)
        print(output)
        return output