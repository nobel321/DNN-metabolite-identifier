# BASIC IMPORTS
import os
import pyodbc
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib as plt
import mysql.connector
import IPython

from rdkit import Chem
from rdkit.Chem import AllChem

from IPython.display import display

# MODEL SPECIFIC IMPORTS
from tensorflow import keras
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# connect to MySQL database
cnx = mysql.connector.connect(
    user='root', 
    password='P0s!ed0n', 
    host='localhost', 
    database='massbank'
    )

# SQL query to select relevant columns from the compound table
compound_query = "SELECT CH_FORMULA, CH_EXACT_MASS, CH_EXACT_MASS_SIGNIFICANT FROM compound"
compound_df = pd.read_sql(compound_query, cnx)

# drop empty rows
compound_df.dropna(inplace=True)

X = compound_df[['CH_EXACT_MASS', 'CH_EXACT_MASS_SIGNIFICANT']].values.astype(float)

# target variable (to be predicted)
y = compound_df['CH_FORMULA'].values

print(X.shape)
print(y.shape)

display(X)
display(y)

# DIVIDE THE DATA AS TRAIN/TEST DATASET
'''
cell_df ➡ Train/Test
Train(X, y) ➡ X is a 2D array
Test(X, y) ➡ y is a 1D array
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

unique_train_labels = np.unique(y_train)
unique_test_labels = np.unique(y_test)

unseen_labels = np.setdiff1d(unique_test_labels, unique_train_labels)

print('Unseen labels:', unseen_labels)

# Find the indices of rows that contain previously unseen labels in both training and test sets
train_mask = np.isin(y_train, unseen_labels, invert=True)
test_mask = np.isin(y_test, unseen_labels, invert=True)

# Filter the data using the masks
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

# Preprocess the filtered data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

label_encoder = LabelEncoder()
num_classes = len(np.unique(y_train_filtered))

y_train_encoded = to_categorical(label_encoder.fit_transform(y_train_filtered), num_classes=num_classes)
y_test_encoded = label_encoder.transform(y_test_filtered)

# define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])

# compile the model
model.compile(optimizer=tf.optimizers.Adam(0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# define early stopping criteria
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# train the model
history = model.fit(X_train_scaled, y_train_encoded, batch_size=32, epochs=10, validation_split=0.2)

# encode test labels to integers

# convert encoded labels to one-hot encoding
y_test_onehot = to_categorical(y_test_encoded, num_classes=num_classes)

# make predictions on test set
y_pred = model.predict(X_test_scaled)
y_pred_class = np.argmax(y_pred, axis=1)

# decode encoded labels
y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
y_pred_decoded = label_encoder.inverse_transform(y_pred_class)

# evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_onehot)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# evaluate the model on the train set
# train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train_encoded)
# print(f'Train Loss: {train_loss}, Train Accuracy: {train_accuracy}')

# generate classification report
print(classification_report(y_test_decoded, y_pred_decoded))

# close the connection to the database
cnx.close()