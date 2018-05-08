#2018-5.4多层感知机
import tensorflow as tf
import numpy as np
class muti_dense(object):
    def __init__(self,
                 num_classes,
                 hidden_dim_1,
                 hidden_dim_2,
                 
                 vocab_size
                 ):
                 
        self.input_x=tf.placeholder(tf.float32,[None,vocab_size],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        def multilayer_perceptron(input_x):
            '''
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            #layer_1 = tf.nn.dropout(layer_1, keep_prob=0.8)  
            #
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            #layer_2=tf.nn.dropout(layer_2,keep_prob=1)
            
            
            # Output layer with linear activation
            out_layer_multiplication = tf.matmul(layer_2, weights['out'])
            y = out_layer_multiplication + biases['out_b']
            '''
            layer_1 = tf.layers.dense(input_x, vocab_size, tf.nn.relu)
            layer_2 = tf.layers.dense(layer_1, hidden_dim_1, tf.nn.relu)
            layer_3 = tf.layers.dense(layer_2,hidden_dim_2, tf.nn.relu)
            y = tf.layers.dense(layer_3, num_classes)
            
            #out_layer = tf.add(tf.matmul(layer_2, weights['out']) , biases['out_b'])
            return y

        weights={'h1':tf.Variable(tf.random_normal([vocab_size,hidden_dim_1])),
                'h2':tf.Variable(tf.random_normal([hidden_dim_1,hidden_dim_2])),
                'out':tf.Variable(tf.random_normal([hidden_dim_2,num_classes]))
       }
        biases={
            'b1':tf.Variable(tf.random_normal([hidden_dim_1])),
            'b2':tf.Variable(tf.random_normal([hidden_dim_2])),
            'out_b':tf.Variable(tf.random_normal([num_classes]))
        }
        
        out_layer=multilayer_perceptron(self.input_x)
                 
        #==========================================
        #输出
        with tf.name_scope("output"):
            self.scores=out_layer
            self.predictions = tf.argmax(self.scores, 1, name="predictions")#预测的类别

        #========================================
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
       # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
                 
        
