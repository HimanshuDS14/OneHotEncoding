import pandas as pd
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv("credit_Score.csv")
print(data.head(10))


ohe = OneHotEncoder(categorical_features=[0])

data = ohe.fit_transform(data).toarray()

print(data)
