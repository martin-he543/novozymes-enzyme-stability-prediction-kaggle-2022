import datetime
import pandas as pd
import numpy as np
import torch
import random
import pickle
from math import isnan, log

class GenerateEmbeddings:
    def __init__(self, save_dir, num_tokens, d_model):
        np.random.seed(0)
        embeddings = np.random.randn(num_tokens, d_model)
        np.save(save_dir,embeddings)
        #'tokens{:%Y.%m.%d.%H.%M.%S}.npy'.format(datetime.datetime.now())

class GenerateData:
    def __init__(self, seq_len, d_model, batch_size, save_dir, n_test_batches=10):
        self.seq_len = seq_len
        self.d_model = d_model
        self.batch_size = batch_size
        self.num_tokens = 20
        self.n_test_batches = n_test_batches
        log_m = 3.919393034783877
        log_d = 0.21659666611061698
        
        #read train.csv and apply updates
        df_train = pd.read_csv("./data/train.csv", index_col="seq_id")
        df_train_updates = pd.read_csv("./data/train_updates_20220929.csv", index_col="seq_id")

        all_features_nan = df_train_updates.isnull().all("columns")

        drop_indices = df_train_updates[all_features_nan].index
        df_train = df_train.drop(index=drop_indices)

        swap_ph_tm_indices = df_train_updates[~all_features_nan].index
        df_train.loc[swap_ph_tm_indices, ["pH", "tm"]] = df_train_updates.loc[swap_ph_tm_indices, ["pH", "tm"]]
        
        #drop sequences longer than seq_len
        df_train.drop(df_train[[len(x) > seq_len for x in df_train.protein_sequence]].index, inplace=True)
        
        #remove duplicates
        seq = df_train["protein_sequence"].values
        pH = df_train["pH"].values
        tm = df_train["tm"].values
        unique_set = set()
        for i, s in enumerate(seq):
            #remove nans
            if isnan(pH[i]):
                pH[i] = 7.0
            unique_set.add((s, pH[i]/7 - 1, (log(tm[i])-log_m)/log_d))
        unique_list = list(unique_set)
        self.n_train_batches = len(unique_list)//self.batch_size - self.n_test_batches
        print(f"Train batches: {self.n_train_batches}")
        print(f"Test batches: {self.n_test_batches}")
        
        #split into train and test
        random.Random(7).shuffle(unique_list)
        data = {
            "train": unique_list[0:self.n_train_batches*self.batch_size],
            "test": unique_list[self.n_train_batches*self.batch_size:]
        }
        with open(save_dir, "wb") as file:
            pickle.dump(data, file)
        
class DataLoader:
    def __init__(self, seq_len, d_model, batch_size, embedding_dir, data_dir):
        self.embeddings = np.load(embedding_dir)
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.batch_size = batch_size
        self.num_tokens = 20
        
        self.train_current = 0
        self.test_current = 0
        
        with open("amino_ranking.txt", "r") as f:
            self.amino_codes = f.read().split("\n")
        
        with open(data_dir, "rb") as file:
            data = pickle.load(file)
        
        self.train = data["train"]
        self.test = data["test"]
        self.n_train_batches = len(self.train)//self.batch_size
        self.n_test_batches = len(self.test)//self.batch_size
        
    def get_train_batch(self):
        sequence = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        ph = torch.zeros(self.batch_size, 1, 1)
        tm = torch.zeros(self.batch_size, 1, 1)
        for i in range(self.batch_size):
            train_index = self.batch_size*self.train_current + i
            seq_string = self.train[train_index][0]
            seq = np.array([self.embeddings[self.amino_codes.index(x)] for x in seq_string])
            if len(seq_string) < self.seq_len:
                seq = np.pad(seq, ((0, self.seq_len - len(seq_string)%self.seq_len), (0,0)), "constant")
            sequence[i] = torch.tensor(seq)
            ph[i] = self.train[train_index][1]
            tm[i] = self.train[train_index][2]
        self.train_current += 1
        if self.train_current == self.n_train_batches:
            self.train_current = 0
        return sequence, ph, tm
    
    def get_test_batch(self):
        sequence = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        ph = torch.zeros(self.batch_size, 1, 1)
        tm = torch.zeros(self.batch_size, 1, 1)
        for i in range(self.batch_size):
            test_index = self.batch_size*self.test_current + i
            seq_string = self.test[test_index][0]
            seq = np.array([self.embeddings[self.amino_codes.index(x)] for x in seq_string])
            if len(seq_string) < self.seq_len:
                seq = np.pad(seq, ((0, self.seq_len - len(seq_string)%self.seq_len), (0,0)), "constant")
            sequence[i] = torch.tensor(seq)
            ph[i] = self.test[test_index][1]
            tm[i] = self.test[test_index][2]
        self.test_current += 1
        if self.test_current == self.n_test_batches:
            self.test_current = 0
        return sequence, ph, tm

if __name__ == "__main__":
    #generate = GenerateData(1024, 128, 32, "train-test.pci")
    #loader = DataLoader(1024, 128, 32, "tokens2022.12.23.15.34.42.npy", "train-test.pci")
    print(len(loader.train))
    print(len(loader.test))
