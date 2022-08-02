import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor

from data import DataLoader
from util.metric import print_accuracy

class Trainer():
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

    def train(self):
        self.regr = RandomForestRegressor(n_estimators=1000, 
                                          random_state=42,
                                         )
        self.regr.fit(self.train_features, self.train_labels)

    def predict(self, features):
        pred = self.regr.predict(features)

        threshold = self.train_labels.mean()
        return np.array([0 if i <= threshold else 1 for i in pred])


if __name__ == '__main__':
    # load data
    dataloader = DataLoader('competition-data/train.csv', 'competition-data/test.csv')
    train_features, val_features, train_labels, val_labels =  dataloader.get_train_val_data()
    
    # train
    trainer = Trainer(train_features, train_labels)
    trainer.train()

    # validate
    pred = trainer.predict(val_features)
    print_accuracy(pred, val_labels)