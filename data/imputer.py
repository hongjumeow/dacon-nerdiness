import numpy as np
from sklearn.impute import SimpleImputer

def impute(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer = imputer.fit(data)
    return imputer.transform(data)