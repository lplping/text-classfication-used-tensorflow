
'''
实现意图识别
'''
import tensorflow as tf
import numpy as np
from mlp import muti_dense
from keras.utils import np_utils
import os
import pickle
flags=tf.app.flags
FLAGS=tf.app.flags.FLAGS
#=================================

flags.DEFINE_integer("hidden_dim_2", 128, "hidden_dim_2")
flags.DEFINE_integer("hidden_dim_1", 512, "hidden_dim_1")
flags.DEFINE_integer("hidden_dim_3", 64, "hidden_dim_1")
flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")

flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
def main(_):
    with open('./train_x.txt','r',encoding='utf-8') as fr:
        lines = fr.readlines()
        train = [line.strip() for line in lines]
    with open('./train_y.txt','r',encoding='utf-8')as fr:
        lines = fr.readlines()
        y = np.array([line.strip() for line in lines])
    #####################
    #测试集
    with open('./test_x.txt','r',encoding='utf-8')as fr:
        lines = fr.readlines()
        test= [line.strip() for line in lines]
    with open('./test_y.txt','r',encoding='utf-8') as fr:
        lines = fr.readlines()
        test_y = np.array([line.strip() for line in lines])
    #数据预处理
    print('yyyyyyyyyyyyyyy')
    #print(y.shape[1])
    y_train = np_utils. to_categorical(y)
    y_dev  =  np_utils.to_categorical(test_y )
    
    #=====================================
    
    with open('term.pkl','rb') as tr_file:
        term_dict=pickle.load(tr_file)
    train_X=np.zeros((len(train),len(term_dict)))  #train
    for i in range(len(train))  :
        for term in train[i].split():
            term_index=term_dict[term]
            train_X[i][term_index]+=1
    
    test_X=np.zeros((len(test),len(term_dict)))  #test
    for i in range(len(test))  :
        for term in test[i].split():
            term_index=term_dict[term]
            test_X[i][term_index]+=1
        #==========================================
    #train
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            mlp_c=muti_dense(
                 num_classes=y_train.shape[1],
                 hidden_dim_1=FLAGS.hidden_dim_1,
                 hidden_dim_2=FLAGS.hidden_dim_2,
                
                 vocab_size=len(term_dict)
                 )
            #=============================
            #定义优化器
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(mlp_c.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # Keep track of gradient values and sparsity (optional)
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", 'a'))
            print("Writing to {}\n".format(out_dir))
    
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", mlp_c.loss)
            acc_summary = tf.summary.scalar("accuracy", mlp_c.acc)
    
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  mlp_c.input_x: x_batch,
                  mlp_c.input_y: y_batch,
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, mlp_c.loss, mlp_c.acc],
                    feed_dict)
                print(" step {}, loss {:g}, acc {:g}".format( step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
    
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set 1
                """
                feed_dict = {
                  mlp_c.input_x: x_batch,
                  mlp_c.input_y: y_batch,
                  
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, mlp_c.loss, mlp_c.acc],
                    feed_dict)
                print("step {}, loss {:g}, acc {:g}".format( step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            def get_batches( sources, targets,batch_size):
                '''
                定义生成器，用来获取batch
                '''
                for batch_i in range(0, len(sources)//batch_size+1):
                    start_i = batch_i * batch_size
                    sources_batch = sources[start_i:start_i + batch_size]
                    targets_batch = targets[start_i:start_i + batch_size]
                    yield  sources_batch,targets_batch
                    # Generate batches
           
            # Training loop. For each batch...
            for epoch in  range(FLAGS.num_epochs):
                for batch_i,(x_batch,y_batch) in enumerate (get_batches( train_X,y_train,128)):
                    train_step(x_batch, y_batch)
                print("\nEvaluation:")
                dev_step(test_X, y_dev, writer=dev_summary_writer)
                print("")
                    
                path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                print("Saved model checkpoint to {}\n".format(path))
    
if __name__=='__main__':
    tf.app.run()   