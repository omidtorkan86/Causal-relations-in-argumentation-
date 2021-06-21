import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from brat_parser import get_entities_relations_attributes_groups
torch.manual_seed(123)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from RISparser import readris


def lemmatize_stemming(word):
    lem = WordNetLemmatizer().lemmatize(word)
    return PorterStemmer().stem(lem)


def preprocess(token):
    sp = stopwords.words("english")
    if token not in sp and len(token) > 2:
        sl = lemmatize_stemming(token)
        return sl


class DataProcessor(object):

    def read_agrotech(self):

        directory = "C:/Users/39351/Desktop/Information Retrieval/Information Retrieval/datasets/agrotech(Dataset2)/agrotech"
        entries = []
        for filename in os.listdir(directory):
            if filename.endswith('.ris'):
                with open(os.path.join(directory, filename), 'r',errors='ignore') as bibliography_file:
                    entry = readris(bibliography_file)
                    entries.extend(entry)
        xx = []
        labels = []
        for e in entries:
            xx.append(e)
        x = []
        for e in xx:
            if "abstract" in e:
                x.append(e['abstract'])
        datas = []
        for d in x:
            for s in sent_tokenize(d):
                datas.append(s)
                labels.append([1, 0, 0, 0, 0, 0])
        for i in datas[128:256]:
            print(i)
        return datas[128:256], labels[128:256]


    def read_text(self,is_train_data):

        # read data from files
        directory =  "C:/Users/39351/Desktop/Information Retrieval/Information Retrieval/datasets/BECAUSE-master(Dataset1)/BECAUSE-master"
        sub_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

        entities_all, relations_all, attributes_all, groups_all = [], [], [], []
        for d in sub_folders:
            d_path = os.path.join(directory, d)
            for filename in os.listdir(d_path):
                if filename.endswith('.ann'):
                    entities, relations, attributes, groups = get_entities_relations_attributes_groups(
                        os.path.join(d_path, filename))
                    entities_all.append(entities)
                    relations_all.append(relations)
                    attributes_all.append(attributes)
                    groups_all.append(groups)

        result = {}
        for dictionary in entities_all:
            result.update(dictionary)
        datas, labels = [], []
        for item in result.values():
            datas.append(item.text)
            labels.append(item.type)

        ##  types: {'Argument', 'Note', 'Motivation', 'NonCausal', 'Consequence', 'Purpose'}##

        # encoding lables to integer
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels_int = le.transform(labels)
        last_labels = []
        print(len(labels))
        # encoding integers to OneHot
        for l in labels_int:
            if l == 0:
                last_labels.append([1,0,0,0,0,0])
            elif l == 1:
                last_labels.append([0, 1, 0, 0, 0, 0])
            elif l == 2:
                last_labels.append([0, 0, 1, 0, 0, 0])
            elif l == 3:
                last_labels.append([0, 0, 0, 1, 0, 0])
            elif l == 4:
                last_labels.append([0, 0, 0, 0, 1, 0])
            elif l == 5:
                last_labels.append([0, 0, 0, 0, 0, 1])

        return datas, last_labels

    
    def word_count(self, datas):

        dic = {}
        for data in datas:
            data_list = data.split()
            for word in data_list:
                word = word.lower()
                if(word in dic):
                    dic[word] += 1
                else:
                    dic[word] = 1
        word_count_sorted = sorted(dic.items(), key=lambda item:item[1], reverse=True)
        return  word_count_sorted
    
    def word_index(self, datas, vocab_size):

        word_count_sorted = self.word_count(datas)
        word2index = {}

        word2index["<unk>"] = 0
        #padding
        word2index["<pad>"] = 1
        

        vocab_size = min(len(word_count_sorted), vocab_size)
        for i in range(vocab_size):
            word = word_count_sorted[i][0]
            word2index[word] = i + 2
          
        return word2index, vocab_size
    
    def get_datasets(self, vocab_size, embedding_size, max_len):

        datas, labels = self.read_text(is_train_data=True)
        train_datas, test_datas,train_labels, test_labels = train_test_split(datas,labels, test_size=.25)
        word2index, vocab_size = self.word_index(train_datas, vocab_size)
        agro_datas,agro_labels = self.read_agrotech()
        #test_datas, test_labels = datas[c+1:],labels[c+1:]
        
        train_features = []
        for data in train_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower()
                #sp = stopwords.words("english")
                #if word not in sp:
                #    word = lemmatize_stemming(word)
                #else:
                #    continue
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])
                if(len(feature)==max_len):
                    break

            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            train_features.append(feature)
            
        test_features = []
        for data in test_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower()
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])
                if(len(feature)==max_len):
                    break

            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            test_features.append(feature)

        agro_features = []
        for data in agro_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower()
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])
                if (len(feature) == max_len):
                    break

            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            agro_features.append(feature)

        print(train_labels)
        train_features = torch.LongTensor(train_features)
        train_labels = torch.FloatTensor(train_labels)

        agro_features = torch.LongTensor(agro_features)
        agro_labels = torch.FloatTensor(agro_labels)

        test_features = torch.LongTensor(test_features)
        test_labels = torch.FloatTensor(test_labels)
        

        embed = nn.Embedding(vocab_size + 2, embedding_size)
        train_features = embed(train_features)
        test_features = embed(test_features)

        agro_features = embed(agro_features)

        train_features = Variable(train_features, requires_grad=False)
        print(len(train_features))
        print(len(train_labels))
        train_datasets = torch.utils.data.TensorDataset(train_features, train_labels)
        
        test_features = Variable(test_features, requires_grad=False)
        test_datasets = torch.utils.data.TensorDataset(test_features, test_labels)

        agro_features = Variable(agro_features, requires_grad=False)
        agro_datasets = torch.utils.data.TensorDataset(agro_features, agro_labels)

        return train_datasets, test_datasets, agro_datasets
    