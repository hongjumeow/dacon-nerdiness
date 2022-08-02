import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

def one_hot_encode(data, val):
    transformer = make_column_transformer(
        (OneHotEncoder(), [val]),
        remainder='passthrough'
    )
    transformed = transformer.fit_transform(data)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names())

    return transformed_df
