import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score

def OneHotEncodeData(data):
    onehot_encoder = OneHotEncoder()

    # country
    onehot = pd.DataFrame(onehot_encoder.fit_transform(data['country'].values.reshape(-1, 1)))
    # print(onehot)
    data = pd.concat([data, onehot], axis=1)
    data = data.drop(['country'], axis=1)
    
    print(data.head(0).columns)
    return data

def Impute(train_feature, test_fetaure):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer = imputer.fit(train_features)
    train_x_imputed = imputer.transform(train_features)
    test_x_imputed = imputer.transform(test_features)

    return train_x_imputed, test_x_imputed

train = pd.read_csv('competition-data/train.csv')
test = pd.read_csv('competition-data/test.csv')

train = train.drop(['index'], axis=1)
test = test.drop(['index'], axis=1)

train_x = train.drop(['nerdiness'], axis=1)
train_x = OneHotEncodeData(train_x)
train_y = train['nerdiness']

index = train_x.head(0).columns

shape = (68, 15000)
data = np.array(train_x)
actual_nerdiness = np.array(train_y)

# print(data[:, 26]) # see country labels 

train_features, test_features, train_labels, test_labels = train_test_split(data, train_y, test_size=0.25)
train_features_imputed, test_features_imputed = Impute(train_features, test_features)


# train

regr = RandomForestRegressor(n_estimators=1000, random_state=42)
regr.fit(train_features_imputed, train_labels)

y_pred = regr.predict(test_features_imputed)

y_pred_bi = np.array([0 if i <= train_labels.mean() else 1 for i in y_pred])
print(y_pred_bi)

errors = abs(y_pred - test_labels)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)
acc = 100 - np.mean(mape)
print('Accuracy:', round(acc, 2), '%.')

acc_score = accuracy_score(y_pred_bi, test_labels)
print(acc_score)
prec_score = precision_score(y_pred_bi, test_labels)
print(prec_score)