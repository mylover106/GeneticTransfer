# coding: utf-8

import gensim
import random
import sys
import getopt
import jieba
from pprint import pprint
from torch.autograd import Variable
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from word import word_plot



class StyleData(object):
    # noinspection SpellCheckingInspection
    def __init__(self, data=[]):
        self.word2index = {}
        self.index2word = {0: 'Eos', 1: 'Sos'}
        self.n_words = 2
        self.word2count = {}
        self.target_style = 1
        for stype in data:
            for seq in stype:
                self.addSequence(seq)

    def addSequence(self, seq):
        for word in seq:
            self.addWord(word)

    def addWord(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2count[word] = 1
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def save(self, name):
        narray = np.array([self.word2index, self.index2word, self.n_words, self.word2count, self.target_style])
        np.save(name, narray)

    def load(self, name):
        narray = np.load(name)  # attention
        self.word2index = narray[0]
        self.index2word = narray[1]
        self.n_words = narray[2]
        self.word2count = narray[3]
        self.target_style = narray[4]

class DsModel(nn.Module):
    """
    notes:
        This model can also be called classfier
    """

    def __init__(self, kind_filters, num_filters, num_in_channels, embedded_size, hidden_size=128):
        """
        Argus:
        kind_filters is a list
        num_filters is the number of filters we want use
        num_in_channels in this case is the number of kinds of embedding
        embedded_size is the embedding size (easy)
        hidden_size = is the hidden_units' number we want to use
        
        Notice:
        kind_filters need to be a list.
        for instance, [1, 2, 3] represent the there are three kind of
        window which's size is 1 or 2 or 3
        the Ds have multi-filter-size and muti-convs-maps
        """
        super(DsModel, self).__init__()

        self.kind_filters = kind_filters
        self.num_filters = num_filters

        self.convs = nn.ModuleList([])
        for width in self.kind_filters:
            self.convs.append(nn.Conv2d(num_in_channels, num_filters, (width, embedded_size)))

        self.linear = nn.Linear(num_filters * len(kind_filters), hidden_size)
        self.linear_out = nn.Linear(hidden_size, 2)
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        """
        this model's inputs should like this N_batch*C_channel*Seq_length*Embedded_size
        if we just use one kind embedding (dynamic or static) the C_channel is 1
        if we use two kind of embedding (dynamic and static) the C_channel is 2
        
        the outputs is the probability of x1 < X1
        """
        convs_outputs = []
        for convs in self.convs:
            convs_outputs.append(convs(x))

        max_pools_outputs = []
        for outputs in convs_outputs:
            max_pools_outputs.append(F.max_pool2d(outputs, kernel_size=(outputs.size()[2], 1)))
            # [2] is the size of high

        flatten = torch.cat(max_pools_outputs, dim=1).view(x.size()[0], -1)
        return self.softmax(self.relu(self.linear_out(self.drop(self.relu(self.linear(flatten))))))

class Embed(nn.Module):
    """
    this is the embedding layer which could embed the index and one-hot logit vector
    but you should indicator use_one_hot or not with index = {True, False}
    """

    def __init__(self, n_vocab, embedding_size):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_size)

    def forward(self, x, index=True):
        if index:
            return self.embedding(x)
        else:
            return torch.mm(x, self.embedding.weight)

class Constants(object):
    def __init__(self, n_vocab):
        self.Lr = 0.0001
        self.Embedding_size = 250
        self.Content_represent = 250
        self.Style_represent = 500
        self.Ey_filters = [1, 2, 3, 4, 5]
        self.Ey_num_filters = 100
        self.D_filters = [2, 3, 4, 5, 6]
        self.D_num_filters = 100
        self.Ds_filters = [1, 2, 3, 4]
        self.Ds_num_filters = 100
        self.Hidden_size = 248
        self.N_vocab = n_vocab
        self.Temper = 0.0001
        self.Max_len = 40
        self.Min_len = 6 # 6 is the max window size of the filters

def eval_func(DNA):
    seq = DNA_to_Words(DNA)
    seq = [style.word2index[word] for word in seq]
    for word in virtul_index:
        seq[word[1]:word[1]] = [style.word2index[word[0]]]
    # build to variable 
    seq_tensor = torch.LongTensor(seq)
    embed = emb[model_index](seq_tensor).unsqueeze(0).unsqueeze(0)
    return ds[model_index](embed).data.numpy()[0][0]

def initDNA(length, number):
    # length is the seq's length of DNA
    # number is the DNAs' number
    # the format is like this 
    # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    
    DNAs = [
        [i] * length for i in range(number)
    ]
    return DNAs

def select(DNAs, number):
    # pass the original DNAs
    DNAs.sort(key=eval_func, reverse=True)
    return DNAs[:10]

def delete_same(DNAs):
    # this function must pass the list not the list copied
    
    for i in range(len(DNAs)-1, -1, -1):
        if i != DNAs.index(DNAs[i]):
            del DNAs[i]

def crossOver(DNAs):
    newDNAs = []
    length = len(DNAs[0]) - 1
    
    for i in range(len(DNAs)):
        for j in range(i+1, len(DNAs)):
            split = random.randint(1, length)
            newDNA = DNAs[i][:split] + DNAs[j][split:]
            newDNAs.append(newDNA)
    
    return newDNAs

def mutualize(DNA, number, prob):
    # this function must pass the list be copied
    
    for i in range(len(DNA)):
        if random.random() <= prob:
            DNA[i] = random.randint(0, number-1)

    return DNA

def batch_mutualize(DNAs, number, prob):
    newDNAs = [mutualize(DNA[:], number, prob) for DNA in DNAs]
    return newDNAs

def DNA_to_Words(DNA):
    seq = []
    for i in range(len(DNA)):
        seq += [wordMap[i][DNA[i]]]
    return seq

def print2seq(DNA):
    seq = DNA_to_Words(DNA)
    for word in virtul_index:
        seq[word[1]:word[1]] = [word[0]]
    return seq

def generate(iter_num, seq_len, number):
    DNAs = initDNA(seq_len, number)
    
    for i in range(iter_num):
        DNAs += batch_mutualize(DNAs, number, 0.5) + crossOver(DNAs)
        delete_same(DNAs)
        DNAs = select(DNAs, 10)
        
    return DNAs


wordvec = gensim.models.KeyedVectors.load_word2vec_format('word2vec_new.txt', binary=False)
style = StyleData()
style.load("./full_style.npy")
const = Constants(style.n_words)
ds = []
emb = []
for i in range(4):
    ds.append(DsModel(embedded_size=const.Embedding_size,
         num_in_channels=1,
         hidden_size=const.Hidden_size,
         kind_filters=const.Ds_filters,
         num_filters=const.Ds_num_filters))
    emb.append(Embed(embedding_size=const.Embedding_size, n_vocab=const.N_vocab))

names = ['dongyeguiwu', 'dongyeguiwu', 'dongyeguiwu', 'wuxia']
for i in range(4):
    ds[i].load_state_dict(torch.load('./Ds_param_' + names[i] + '.pkl', map_location=lambda storage, loc: storage))
    emb[i].load_state_dict(torch.load('./embedding_param_' + names[i] + '.pkl', map_location=lambda storage, loc: storage))



wordMap = None
virtul_words = ['是', '个', '你', '它', '又', '让', '的', '为','吗', '我']
virtul_index = None
symbols = ['.', ',', '?', '？', '。', '，', '、', ':', '：', '!', '!']
model_index = None
def main(seq_example=None, index=None):
    global virtul_index
    global wordMap
    global model_index
    model_index = index
    if seq_example is None:
        seq_example = "真好,这个苹果真好吃，你还有吗？我想再吃一个。"
    seq = jieba.cut(seq_example)
    seq = [word for word in seq]
    symbol_index = []
    for i, word in enumerate(seq):
        if word in symbols:
            symbol_index.append((word, i))

    for i in range(len(seq)-1, -1, -1):
        if seq[i] in symbols:
            del seq[i]

    try:
        wordMap = [
            [x[0] for x in [(word, 0)] + wordvec.most_similar(word, topn=20)] for word in seq if word not in virtul_words
        ]
    except:
        print("The Word isn't in the List, Error")
        return None
    
    virtul_index = []  # save tuple data
    for i, word in enumerate(seq):
        if word in virtul_words:
            virtul_index.append((word, i))

    ans = generate(50,len(wordMap),15)
    for x in ans:
        print(eval_func(x))
    # for i in range(len(ans)):
    #     for symbol in symbol_index:
    #         ans[i][symbol[1]:symbol[1]] = [symbol[0]]

    # for x in ans:
    #     print(''.join(print2seq(x)))

    # for x in ans:
    #     print(x)
    print_seqs = []
    for x in ans:
        print_seqs.append(print2seq(x))

    for i, seq in enumerate(print_seqs):
        word_plot(words="".join(seq), save_file=str(i) + 'new.png')
    

    
    for i in range(len(print_seqs)):
        for symbol in symbol_index:
            print_seqs[i][symbol[1]:symbol[1]] = [symbol[0]]


    # for x in print_seqs:
    #     print(''.join(x))

    print_seqs = [''.join(x) for x in print_seqs]
    for seq in print_seqs:
        print(seq)

    return print_seqs

if __name__ == "__main__":
    main(index = 1)
