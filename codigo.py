#1- IMPORTAR LIBRERÍAS

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


from google.colab import files
uploaded = files.upload()

true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

#para unir los datasets, el Label
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)

df.head(5)

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['title']).toarray()
y = df['label'].values
