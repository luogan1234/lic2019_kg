import tensorflow as tf
import paras
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def add_embedding_layer(word_indexes, pos_indexes):
    word_embedding = tf.truncated_normal([paras.DISTINCT_WORDS, paras.WORD_EMBEDDING_SIZE], stddev=0.1)
    pos_embedding = tf.truncated_normal([paras.DISTINCT_POS, paras.POS_EMBEDDING_SIZE], stddev=0.1)
    e1 = tf.nn.embedding_lookup(word_embedding, word_indexes)
    e2 = tf.nn.embedding_lookup(pos_embedding, pos_indexes)
    output = tf.concat([e1, e2], 2)
    output = tf.nn.dropout(output, keep_prob=keep_prob)
    return output

def add_bidirectional_lstm(inputs, fw_unit, bw_unit, i):
    with tf.variable_scope('lstm_'+i):
        cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(fw_unit, dropout_keep_prob=keep_prob)
        cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(bw_unit, dropout_keep_prob=keep_prob)
        outputs, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32)
        output = tf.concat(outputs, 2)
    return output

def get_cnn_features(inputs, l, r, succ):
    with tf.variable_scope('cnn_'+succ):
        ele = (inputs, l, r)
        input_slice = tf.map_fn(lambda x: tf.concat([tf.zeros([x[1], paras.VEC]), x[0][x[1]:x[2], :], tf.zeros([paras.MAX_LEN-x[2], paras.VEC])], 0), ele, tf.float32)
        output = tf.layers.conv1d(input_slice, paras.FILTERS, paras.KSIZE, padding='same', activation=tf.nn.relu)
        output = tf.reshape(output, [-1, paras.FILTERS*paras.MAX_LEN])
        output = tf.layers.dense(output, paras.MAX_LEN)
        output = tf.nn.dropout(output, keep_prob=keep_prob)
    return output

def get_segment_features(inputs, pos):
    ele = (inputs, pos)
    output = tf.map_fn(lambda x: tf.zeros([paras.VEC]) if x[1] not in range(0, paras.MAX_LEN) else x[0][x[1]], ele, tf.float32)
    return output

def get_dist_features(d):
    output = tf.one_hot(d, paras.MAX_LEN)
    return output

def add_fc_layer(inputs, num, act):
    output = tf.layers.dense(inputs, num, kernel_initializer=tf.truncated_normal_initializer(), activation=act)
    return output

word_indexes = tf.placeholder(tf.int32, [None, None])
pos_indexes = tf.placeholder(tf.int32, [None, None])
seq_length = tf.placeholder(tf.int32, [None])
mask = tf.sequence_mask(seq_length, dtype=tf.float32)
outputs = tf.placeholder(tf.int32, [None, None])
relation_data = tf.placeholder(tf.int32, [None, 10])  # s_l, s_r, o_l, o_r, m_l, m_r, left, right, d, r_type
pos_weight = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False, name='global_step')

inputs = add_embedding_layer(word_indexes, pos_indexes)
lstm1 = add_bidirectional_lstm(inputs, paras.FW_UNIT, paras.BW_UNIT, '0')
lstm2 = add_bidirectional_lstm(lstm1, paras.FW_UNIT, paras.BW_UNIT, '1')
lstm3 = add_bidirectional_lstm(lstm2, paras.FW_UNIT, paras.BW_UNIT, '2')
logits1 = add_fc_layer(lstm3, 5, None)

#raw_loss1 = tf.nn.weighted_cross_entropy_with_logits(targets=outputs, logits=logits1, pos_weight=pos_weight)
#loss1 = tf.reduce_mean(raw_loss1) / (1 + (pos_weight - 1) / (pos_weight + 1))
#logits1_label = tf.round(tf.sigmoid(logits1))
with tf.name_scope('loss1'):
    loss1 = tf.contrib.seq2seq.sequence_loss(targets=outputs, logits=logits1, weights=mask)
    logits1_label = tf.argmax(logits1, 2)
    tf.summary.scalar('loss1', loss1)
    tf.summary.histogram('logits1_label', logits1_label)

relation_inputs = tf.concat([lstm3, logits1], 2)
fs = get_cnn_features(relation_inputs, relation_data[:, 0], relation_data[:, 1], 'sub')
fo = get_cnn_features(relation_inputs, relation_data[:, 2], relation_data[:, 3], 'ob')
fm = get_cnn_features(relation_inputs, relation_data[:, 4], relation_data[:, 5], 'mid')
fl = get_segment_features(relation_inputs, relation_data[:, -4])
fr = get_segment_features(relation_inputs, relation_data[:, -3])
fd = get_dist_features(relation_data[:, -2])
f_concat = tf.concat([fs, fo, fm, fl, fr, fd], 1)
logits2 = add_fc_layer(f_concat, paras.SCHEMA_NUMBER, None)

with tf.name_scope('loss2'):
    raw_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=relation_data[:, -1], logits=logits2)
    loss2 = tf.reduce_mean(raw_loss2)
    logits2_label = tf.argmax(logits2, 1)
    tf.summary.scalar('loss2', loss2)
    tf.summary.histogram('logits2_label', logits2_label)

loss = loss1 + loss2

with tf.name_scope('train'):
    train_func = tf.train.AdamOptimizer().minimize(loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)

def init():
    if tf.gfile.Exists(paras.LOG):
        tf.gfile.DeleteRecursively(paras.LOG)
    tf.gfile.MakeDirs(paras.LOG)

def train(train_data):
    min_loss = 0
    max_acc = 0
    global_step = 0
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(paras.LOG + '/train', sess.graph)
        if os.path.exists('tmp/kg.meta'):
            saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
        else:
            tf.global_variables_initializer().run()
        for epoch in range(10):
            epoch_loss = 0.0
            epoch_acc = 0.0
            train_data.init_batch()
            calc_step = 0
            print('epoch %d:' % epoch)
            STEP = (train_data.num - 1) // paras.BATCH_SIZE + 1
            for step in range(1, STEP):
                bd = train_data.next_batch(paras.BATCH_SIZE)
                #Y = np.array(bd.output).flatten()
                #w = min(len(Y) / np.sum(Y) - 1.0, 16.0)
                feed_dict = {word_indexes: bd.index, pos_indexes: bd.pos, seq_length: bd.seq_length, outputs: bd.output, relation_data: bd.relation_data, is_training: True, keep_prob: 0.7}
                summary, _ = sess.run([merged, train_func], feed_dict=feed_dict)
                global_step += 1
                train_writer.add_summary(summary, global_step)
                print('step:', step, 'batch max len:', bd.max_len)
                if step % 20 == 0:
                    feed_dict = {word_indexes: bd.index, pos_indexes: bd.pos, seq_length: bd.seq_length, outputs: bd.output, relation_data: bd.relation_data, is_training: False, keep_prob: 1.0}
                    batch_loss1 = sess.run(loss1, feed_dict=feed_dict)
                    batch_loss2 = sess.run(loss2, feed_dict=feed_dict)
                    batch_loss = batch_loss1 + batch_loss2
                    _Y1 = sess.run(outputs, feed_dict=feed_dict)
                    _Y2 = sess.run(logits1_label, feed_dict=feed_dict)
                    Y1, Y2 = [], []
                    for i in range(len(bd.seq_length)):
                        Y1.extend(_Y1[i][:bd.seq_length[i]])
                        Y2.extend(_Y2[i][:bd.seq_length[i]])
                    Y3 = np.array(bd.relation_data)[:, -1].flatten()
                    Y4 = sess.run(logits2_label, feed_dict=feed_dict)
                    Y4 = np.array(Y4).flatten()
                    epoch_loss += batch_loss
                    calc_step += 1
                    acc = accuracy_score(Y3, Y4)
                    epoch_acc += acc
                    print('progress: %.3lf batch_loss1: %.5lf batch_loss2: %.5lf' % (step / STEP, batch_loss1, batch_loss2))
                    print('accuracy:', acc)
                    #print('true label:', Y3)
                    #print('pred label:', Y4)
                    print(classification_report(Y1, Y2, target_names=['0', '1', '2', '3', '4']))
            mean_loss = epoch_loss / calc_step
            mean_acc = epoch_acc / calc_step
            print('epoch:', epoch, 'mean loss:', mean_loss, 'mean acc:', mean_acc)
            print('---------------------------------')
            if mean_acc > max_acc:
                max_acc = mean_acc
                saver.save(sess, 'tmp/kg')
        train_writer.close()

def predict(test_data):
    with tf.Session() as sess:
        if os.path.exists('tmp/kg.meta'):
            saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
        now = 0
        test_data.i = 0
        for step in range(0, test_data.num // paras.BATCH_SIZE + 1):
            bd = test_data.next_batch(paras.BATCH_SIZE)
            feed_dict = {word_indexes: bd.index, pos_indexes: bd.pos, is_training: False, keep_prob: 1.0}
            logit_res = sess.run(logits_label, feed_dict=feed_dict)
            print(logit_res)
            print('predicting batch number:', step)
            for one in logit_res:
                test_data.data[now].output_to_spo_list(one)
                now += 1
                if now >= test_data.num:
                    return
    assert EOFError