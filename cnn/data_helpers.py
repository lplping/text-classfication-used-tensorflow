import numpy as np
import pandas as pd
import codecs
from keras.preprocessing.sequence import pad_sequences
import pickle
#

#加载数据
def load_data_and_labels(filepath):
    data=pd.read_excel(filepath,header=None)
    data['sent']=data[0]
    data['label']=data[2]
    text=data['sent']
    label=data['label']
    return text,label
#加载数据
def load_test_and_labels(filepath):
    data=pd.read_excel(filepath,header=None)
    data['sent']=data[0]
    data['label']=data[1]
    text=data['sent']
    label=data['label']
    return text,label
#
def build_train_data(label_file,filepath):
    #加载标签数据
    label_vob={}
    for line in codecs.open(label_file,'r','utf-8'):
        name=line.split(',')[0]
        label_vob[name]=line.split(',')[1]
    text,label=load_data_and_labels(filepath)
    labels=[int(label_vob[line]) for line in label]
    return text,labels
#=======================
#转化为索引
def cret_dict(data):
    codes=['<PAD>','<UNK>']
    set_words=set([term for line in data for term in line])
    #=========================================================
    #存储所有字的文件,以便测试加载
    #pickle.dump(set_words,open('./vocab.pkl','wb'))
                
    int_to_vab={word_i:word for word_i,word in enumerate(codes+list(set_words))}
    vab_to_int={word:word_i for word_i,word in int_to_vab.items()}
    return int_to_vab,vab_to_int
    

def pad_sentences(vab_int,max_len):
    #首先把句子长度不成max_len+20,如果句子长度不足max_len,那么把句子尾部补0
    data=pad_sequences(vab_int,max_len,padding='post')
    return data

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
