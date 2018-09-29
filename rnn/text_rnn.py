#
#预训练的词向量
import tensorflow as tf
class TextRNN(object):
    
    def __init__(self, sequence_length, num_classes, vocab_size,batch_size,
      embedding_size, hidden_size, num_layers):
        #==============================
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='drop_keep_prob')
        
        #=============================
        #embedding_layer
        with tf.name_scope('embedding'):
            # vocab size * hidden size, 将单词转成embedding描述
            embeding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32) 
            embed_input=tf.nn.embedding_lookup(embeding,self.input_x)
            #embeding=word_embedding_matrix
            #embed_input=tf.nn.embedding_lookup(embeding,self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(embed_input, -1)
        #====================================
        #======================================
        #搭建LSTM 模型
        def cell():
            #定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            lstmCell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)
            #添加 dropout layer, 一般只设置 output_keep_prob
            lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.dropout_keep_prob)
            return lstmCell
        #value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        #调用 MultiRNNCell 来实现多层 LSTM
        with tf.name_scope("rnn"):
            # 多层rnn网络
            rnn_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(num_layers)], state_is_tuple=True)
            #cells = [[lstmCell] for _ in range(num_layers)]
            # **步骤5：用全零来初始化state
            init_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            #rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            #步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embed_input, dtype=tf.float32)
            
            #_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embed_input, initial_state=init_state,time_major=False)
            #last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            last=tf.reduce_mean(_outputs,axis=1)#取平均
            print('last.shape:',last.shape)
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, hidden_size, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)
            print('x1.shape:',fc.shape)
            # 分类器
            self.scores = tf.layers.dense(fc, num_classes, name='fc2')
            print('x2.shape:',self.scores.shape)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
