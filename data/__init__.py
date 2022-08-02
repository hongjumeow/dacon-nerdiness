import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .encoder import one_hot_encode
from .imputer import impute

class DataLoader():
    def __init__(self, path_train, path_test):
        self.trainset = pd.read_csv(path_train)
        self.testset = pd.read_csv(path_test)

        self.trainset = self.trainset.drop(['index'], axis=1)
        self.testset = self.testset.drop(['index'], axis=1)
    
    def split_label(self, dataset, val):
        return dataset.drop([val], axis=1), dataset[val]

    def get_train_val_data(self):
        self.train_x, self.train_y = self.split_label(self.trainset, 'nerdiness')
        self.train_x = one_hot_encode(self.train_x, 'country')

        self.train_x, self.train_y = np.array(self.train_x), np.array(self.train_y)

        return train_test_split(impute(self.train_x), self.train_y, test_size=0.25)

    def get_test_data(self):
        return impute(self.testset)
        
    def get_categories(self):
        return self.trainset.head(0).columns
    
    def get_labels_by_category(self, data, category):
        #return data[:, data.index(category)]
        return
    
    