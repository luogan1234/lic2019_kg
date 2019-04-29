import tensorflow as tf
import paras
import os

def add_embedding_layer(word_indexes, pos_indexes):
    word_embedding = tf.truncated_normal([paras.DISTINCT_WORDS, paras.WORD_EMBEDDING_SIZE], stddev=0.1)
    pos_embedding = tf.truncated_normal([paras.DISTINCT_POS, paras.POS_EMBEDDING_SIZE], stddev=0.1)
    e1 = tf.nn.embedding_lookup(word_embedding, word_indexes)
    e2 = tf.nn.embedding_lookup(pos_embedding, pos_indexes)
    output = tf.concat([e1, e2], 2)
    return output

def add_stacked_bidirectional_lstm(inputs, fw_units, bw_units):
    output = inputs
    for i in range(len(fw_units)):
        with tf.variable_scope('layer_{}'.format(i), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(fw_units[i], initializer=tf.orthogonal_initializer())
            cell_bw = tf.contrib.rnn.LSTMCell(bw_units[i], initializer=tf.orthogonal_initializer())
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)
            outputs, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, dtype=tf.float32)
            output = tf.concat(outputs, 2)
    return output

def add_fc_layer(prev, num):
    dense = tf.layers.dense(prev, num, kernel_initializer=tf.truncated_normal_initializer())
    output = tf.layers.batch_normalization(dense, training=is_training)
    return output

word_indexes = tf.placeholder(tf.int32, [None, paras.MAX_LEN])
pos_indexes = tf.placeholder(tf.int32, [None, paras.MAX_LEN])
entity_label = tf.placeholder(tf.float32, [None, paras.MAX_LEN, paras.ENTITY_TYPES])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False, name='global_step')
learn_rate = tf.train.exponential_decay(1e-4, global_step, 100, 0.9999, staircase=True)

step1_input = add_embedding_layer(word_indexes, pos_indexes)
lstm_output = add_stacked_bidirectional_lstm(step1_input, paras.FW_UNITS, paras.BW_UNITS)
logits1 = add_fc_layer(lstm_output, paras.ENTITY_TYPES)

step1_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=entity_label, logits=logits1)
train_func = tf.train.AdamOptimizer(learn_rate).minimize(step1_loss, global_step=global_step)
saver = tf.train.Saver(max_to_keep=1)

def train(data_collection):
    with tf.Session() as sess:
        if os.path.exists('tmp/kg.meta'):
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
        else:
            tf.global_variables_initializer().run()
        for epoch in range(10):
            epoch_loss = 0.0
            for step in range(1, 301):
                bd = data_collection.train_data.next_batch(paras.BATCH_SIZE)
                feed_dict = {word_indexes: bd.index, pos_indexes: bd.pos, entity_label: bd.entity_label, is_training: True, keep_prob: 0.8}
                sess.run(train_func, feed_dict=feed_dict)
                if step % 50 == 0:
                    feed_dict = {word_indexes: bd.word_indexes, pos_indexes: bd.pos_indexes, entity_label: bd.entity_label, is_training: False, keep_prob: 1.0}
                    batch_loss = sess.run(step1_loss, feed_dict=feed_dict)
                    epoch_loss += batch_loss
                    print('batch_loss:', batch_loss)
            print('---------------------------------')
            lr = sess.run(learn_rate)
            print('epoch:', epoch, 'lr:', lr, 'average loss:', epoch_loss / step)
