#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import pickle
from keras.utils import np_utils

import datetime
import data_helpers
from text_gru import TextGRU
from tensorflow.contrib import learn
# Parameters
# ==================================================

# Data loading params
flags=tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
flags.DEFINE_string("train_file", "./data/lan.xlsx", "Data source for the positive data.")
flags.DEFINE_string("label_file", "./labels/label_name.txt", "Data source for the negative data.")
   # Model Hyperparameters
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_integer("hidden_size", 128, "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_layers", 2, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

   # Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 128)")
flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 必须通过以下方式才可以调用flags解析
'''

def main(_):
    print(FLAGS.test_file)
if __name__=='__main__':
    tf.app.run()
'''
# ==================================================
def main(_):
    # Load data
    print("Loading data...")
    
    x_,y=data_helpers.build_train_data(FLAGS.label_file,FLAGS.train_file)
    train_int_to_vab,train_to_int=data_helpers.cret_dict(x_)
    #保存对应的词和词索引
    
    #存储所有字的文件,以便测试加载
    pickle.dump(train_int_to_vab,open('./vocab_index.pkl','wb'))
    
    train_ids=[[train_to_int.get(term,train_to_int['<UNK>']) for term in line] for line in x_]
    x_=data_helpers.pad_sentences(train_ids,20)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x_[shuffle_indices]
    y=np.array(y)
    y_shuffled = y[shuffle_indices]
    folids_list=data_helpers.cross_validation_split_for_smp(x_shuffled,y_shuffled)
    for i in range(10):
        
        if not os.path.exists('save_model/'+str(i)+'/'):
            os.makedirs(os.path.join('save_model',str(i)))
        else:
            continue
        
        
        
    
    for i in range(10):
        best_acc=0.0
        print(i)
        print('##################')
        x_train,y_train,x_dev,y_dev=folids_list[i]
    
        y_train=np_utils. to_categorical(y_train)
        y_dev=np_utils. to_categorical(y_dev)
    
    # ==================================================

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
             allow_soft_placement=FLAGS.allow_soft_placement,
             log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                rnn = TextGRU(
                sequence_length=x_train.shape[1],
                
                num_classes=y_train.shape[1],
                vocab_size=len(train_int_to_vab),
                batch_size=FLAGS.batch_size,
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                num_layers=FLAGS.num_layers
                #word_embedding_matrix=embeding_matric
                )
    
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(rnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    
               # Initialize all variables
                sess.run(tf.global_variables_initializer())
    
                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                      rnn.input_x: x_batch,
                      rnn.input_y: y_batch,
                      rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, rnn.loss, rnn.accuracy],
                        feed_dict)
                    return step, loss, accuracy
    
                def dev_step(x_batch, y_batch):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                      rnn.input_x: x_batch,
                      rnn.input_y: y_batch,
                      rnn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy = sess.run(
                        [global_step,  rnn.loss, rnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print('dev')
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    return accuracy
                def save_best_model(sess,path):
                    path = saver.save(sess, path)
                   
                for epoch in range(FLAGS.num_epochs):
                    print('epoch',epoch)
                    # Generate batches 
                    for batch_i,(x_batch, y_batch) in enumerate(data_helpers.get_batches(y_train, x_train, FLAGS.batch_size)):
                       
                        step, train_loss, train_accuracy=train_step(x_batch, y_batch)
                        #print('step',step)
                        if batch_i % FLAGS.evaluate_every == 0:
                            time_str = datetime.datetime.now().isoformat()
                            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, train_loss, train_accuracy))
                    
                         #=====================
                    accuracy=dev_step(x_dev, y_dev)
                    if accuracy>best_acc:
                        best_acc=accuracy
                        print('save_model'+str(i)+'/best_model.ckpt')
                        save_best_model(sess,'save_model/'+str(i)+'/best_model.ckpt')

if __name__=='__main__':
    tf.app.run()
