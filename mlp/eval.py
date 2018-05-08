#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import pickle
# Parameters
# ==================================================

# Data Parameters
# Data loading params
flags=tf.app.flags
FLAGS = tf.app.flags.FLAGS

# Eval Parameters
flags.DEFINE_string("checkpoint_dir", "./runs/a/checkpoints", "Checkpoint directory from training run")
flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# CHANGE THIS: Load data. Load your own data here
def main(_):
    
    with open('./test_x.txt','r',encoding='utf-8')as fr:
        lines = fr.readlines()
        test= [line.strip() for line in lines]
    with open('./test_y.txt','r',encoding='utf-8') as fr:
        lines = fr.readlines()
        y_test =[line.strip() for line in lines]
        y_test =[int(line)for line in lines]

    with open('term.pkl','rb') as tr_file:
        term_dict=pickle.load(tr_file)
    #=====================================
    
    test_X=np.zeros((len(test),len(term_dict)))  #test
    for i in range(len(test))  :
        for term in test[i].split():
            term_index=term_dict[term]
            test_X[i][term_index]+=1
    
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
            input_x = graph.get_operation_by_name("input_x").outputs[0]
    
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            all_predictions = []
            batch_predictions = sess.run(predictions, {input_x: test_X})
                
            all_predictions = np.concatenate([all_predictions, batch_predictions])
    
            #for x_test_batch in batches:
                
    print(all_predictions)
    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    
if __name__=='__main__':
    tf.app.run()