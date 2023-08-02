#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


compound_a = np.loadtxt('compoundA.txt')
substrate = np.loadtxt('substrate.txt')
biomass = np.loadtxt('biomass.txt')

df = pd.DataFrame({'compound_a': compound_a, 'substrate': substrate, 'biomass': biomass})
df


# In[3]:


# plot the data
plt.figure(figsize=(12, 4))
plt.plot(df.compound_a, label='compound A')
plt.plot(df.substrate, label='substrate')
plt.plot(df.biomass, label='biomass')
plt.legend()
plt.show()


# In[4]:


# check data, should be positive
print('compound A:', compound_a.min())
print('substrate:', substrate.min())
print('biomass:', biomass.min())


# In[5]:


# fill abnormal data with 0
df.loc[df.compound_a < 0, 'compound_a'] = 0
df.loc[df.substrate < 0, 'substrate'] = 0
df.loc[df.biomass < 0, 'biomass'] = 0


# In[6]:


# plot the data
plt.figure(figsize=(12, 4))
plt.plot(df.compound_a, label='compound A')
plt.plot(df.substrate, label='substrate')
plt.plot(df.biomass, label='biomass')
plt.legend()
plt.show()


# In[10]:


gen_compound_a = np.loadtxt('gen_compoundA.txt')
gen_substrate = np.loadtxt('gen_substrate.txt')
gen_biomass = np.loadtxt('gen_biomass.txt')

df_test = pd.DataFrame({
    'compound_a': gen_compound_a, 
    'substrate': gen_substrate, 
    'biomass': gen_biomass})

plt.figure(figsize=(12, 4))
plt.plot(df_test.compound_a, label='compound A')
plt.plot(df_test.substrate, label='substrate')
plt.plot(df_test.biomass, label='biomass')
plt.legend()
plt.title('Generalization data')
plt.show()


# In[11]:


df_train = df


# In[12]:


target = 'biomass'

X_train, y_train = df_train.drop(target, axis=1), df_train[target]
X_test, y_test = df_test.drop(target, axis=1), df_test[target]


# In[13]:


# normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[14]:


# use tensorflow keras to build a model


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[109]:


Ni = 2 # number of input features
Nh = 30 # number of hidden units
No = 1 # number of outputs = 1


# In[158]:


model = Sequential()
model.add(Dense(Nh, input_shape=(Ni,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(No, activation='linear'))
model.compile(Adam(lr=0.002), 'mean_squared_error')


# In[159]:


model.summary()


# In[160]:


# train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# use early stopping to prevet overfitting
history = model.fit(X_train, y_train, epochs=200, )


# In[161]:


# plot history
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.title('loss during training')
plt.show()


# In[162]:


# make predictions
y_pred = model.predict(X_test).flatten()


# In[163]:


# plot the results
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label='Real biomass')
plt.plot(y_pred, label='Estimated biomass')
plt.legend()
plt.show()


# In[164]:


def calculate_ia(y_pred, y_true):
    mean = y_true.mean()

    oi = np.abs(y_pred - mean)
    pi = np.abs(y_true - mean)
    ia = 1 - np.sum(np.square(y_pred - y_true)) / np.sum(np.square(oi + pi))
    return ia


def calculate_rms(y_pred, y_true):
    return np.sqrt(
        np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_true))
    )


def calculate_rsd(y_pred, y_true):
    return np.sqrt(
        np.sum(np.square(y_pred - y_true)) / len(y_true)
    )


# In[165]:


print("IA:", calculate_ia(y_pred, y_test.values))
print("RMS:", calculate_rms(y_pred, y_test.values))
print("RSD:", calculate_rsd(y_pred, y_test.values))

