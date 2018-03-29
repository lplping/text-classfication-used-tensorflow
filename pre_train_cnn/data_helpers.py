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
    
#=========================
#预训练词向量
def word2tove(vabb_to_int,embedding_dim,embedding_index):
    nb_words=len(vabb_to_int)
    print('nb_words',nb_words)
    word_embedding_matrix=np.zeros((nb_words,embedding_dim),dtype=np.float32)
    for word ,i in vabb_to_int.items():
        if word in embedding_index:
            word_embedding_matrix[i]=embedding_index[word]
        else:
            new_embedding=np.array(np.random.uniform(-0.5,0.5,embedding_dim))
            embedding_index[word]=new_embedding
            word_embedding_matrix[i]=new_embedding
    print(len(word_embedding_matrix))
    return word_embedding_matrix
def word_matric(vab_to_int):
    
    embeddings_index={}
    with open('wikiw2v_zh_zi.txt','r') as f:
        for line in f:
            line=line.strip()
            arr=line.split(' ')
            embedding=[float(val) for val in arr[1:]]
            word=arr[0]
            embeddings_index[word]=embedding
    print(len(embeddings_index))
    embedding_dim=200
    word_embedding_matrix=word2tove(vab_to_int,embedding_dim,embeddings_index)
    
    print(np.shape(word_embedding_matrix))
    return word_embedding_matrix

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
