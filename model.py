import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump
poly = PolynomialFeatures(degree=2)

df = pd.read_csv('./bodyfat.csv')
df.columns = df.columns.str.strip()
X = poly.fit_transform(df[['weight', 'height', 'neck', 'chest', 'abdomen', 'hip']])

resCol = ['thigh', 'knee', 'ankle', 'biceps', 'forearm', 'wrist']

LR_models = {}

for col in resCol:
  X_train = X.copy()
  y_train = df[col]
  model = LinearRegression()
  model.fit(X_train, y_train)
  LR_models[col] = model

for col, model in LR_models.items():
  model_filename = f'{col}_model.joblib'
  dump(model, f'model_predict/{model_filename}')
