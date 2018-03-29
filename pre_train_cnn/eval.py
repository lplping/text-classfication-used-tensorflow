#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
from keras.utils import np_utils
import datetime
import data_helpers
from text_cnn_pre import TextCNN
from tensorflow.contrib import learn
import csv
import pickle

# Parameters
# ==================================================

# Data Parameters
# Data loading params
flags=tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string("test_file", "./data/test.xlsx", "Data source for the negative data.")
flags.DEFINE_string("label_file", "./data/label_name1.txt", "Data source for the negative data.")

# Eval Parameters
flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
flags.DEFINE_string("checkpoint_dir", "./runs/1521114817/checkpoints", "Checkpoint directory from training run")
flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# CHANGE THIS: Load data. Load your own data here
def main(_):
    text, y_test =data_helpers.load_test_and_labels(FLAGS.test_file)
    with open('vocab_index.pkl','rb') as tr_file:  
        train_int_to_vab=pickle.load(tr_file) 
    #print (train_int_to_vab)
    train_to_int={word:word_i for word_i,word in train_int_to_vab.items()}
    test_ids=[[train_to_int.get(term,train_to_int['<UNK>']) for term in line] for line in text]
    x_test=data_helpers.pad_sentences(test_ids,20)
    print(x_test[:3])
    print("\nEvaluating...\n")
    
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
    
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("drop_keep_prob").outputs[0]
    
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
    
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1)
    
            # Collect the predictions here
            all_predictions = []
            batch_predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
                
            all_predictions = np.concatenate([all_predictions, batch_predictions])
    
            #for x_test_batch in batches:
                
    print(all_predictions)
    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    
    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(text), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
if __name__=='__main__':
    tf.app.run()
