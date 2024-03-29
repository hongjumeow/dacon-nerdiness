import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def LabelEncodeData(data):
    label_encoder = LabelEncoder()

    # country
    data['country'] = label_encoder.fit_transform(data['country'])

    return data

train = pd.read_csv('competition-data/train.csv')
test = pd.read_csv('competition-data/test.csv')

train = train.drop(['index'], axis=1)
test = test.drop(['index'], axis=1)

train_x = train.drop(['nerdiness'], axis=1)
train_x = LabelEncodeData(train_x)
train_y = train['nerdiness']

shape = (68, 15000)
data = np.array(train_x)
actual_nerdiness = np.array(train_y)

train_features, test_features, train_labels, test_labels = train_test_split(data, train_y, test_size=0.25)