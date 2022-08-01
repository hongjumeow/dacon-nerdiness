import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

index = train_x.head(0).columns

shape = (68, 15000)
data = np.array(train_x)
actual_nerdiness = np.array(train_y)


# print(data[:, 26]) # see country labels 
print(len(data[0]))



train_features, test_features, train_labels, test_labels = train_test_split(data, train_y, test_size=0.25)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
train_x_imputed = imputer.fit(train_features)
train_x_imputed = imputer.transform(train_features)
test_x_imputed = imputer.transform(test_features)

regr = RandomForestRegressor()
regr.fit(train_x_imputed, train_labels)

y_pred = regr.predict(test_x_imputed)

y_pred_bi = np.array([0 if i <= train_labels.mean() else 1 for i in y_pred])
print(y_pred_bi)

acc_score = accuracy_score(y_pred_bi, test_labels)
print(acc_score)