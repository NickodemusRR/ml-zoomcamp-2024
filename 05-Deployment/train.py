#!/usr/bin/env python
# coding: utf-8

# # ML Zoomcamp 2024 - Deployment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import auc, roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

# Parameters
C = 1.0
n_splits = 5
output_file = f"model_C={C}.bin"

# 1. Data preparation
df = pd.read_csv("../bank/bank-full.csv", sep=";")
df = df[
    [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    ]
]

df.y = (df.y == "yes").astype(int)

# 2. Dataset splitting
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# 3. Selecting features and target variable
numerical = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
categorical = ["job", "marital", "education", "housing", "contact", "month", "poutcome"]

y_full_train = df_full_train.y.values
y_test = df_test.y.values


# 4. Model validation
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


print(f"Doing validation with C={C}")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.y.values
    y_val = df_val.y.values

    dv, model = train(df_train, y_train, C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"AUC on fold {fold} is {auc.round(3)}")
    fold += 1

print(
    f"C: {C} | AUC mean: {np.mean(scores).round(3)} | AUC std: {np.std(scores).round(3)}"
)


# 5. Training Final Model
print("Training the final model.")

dv, model = train(df_full_train, y_full_train, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print(f"AUC of the final model: {auc.round(3)}")


# 6. Save the model
with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"The model is saved to {output_file}")
