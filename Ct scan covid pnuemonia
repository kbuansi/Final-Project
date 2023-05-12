#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import InputLayer, Lambda, Conv2D, Dropout, MaxPooling2D, Flatten, Dense


# In[2]:


normal_img_path = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\archive (2)\curated_data\curated_data\Cov\NonCovid'
covid_img_path = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\archive (2)\curated_data\curated_data\Cov\Covid'


# In[3]:


normal_img_files = os.listdir(normal_img_path)
covid_img_files = os.listdir(covid_img_path)


# In[4]:


normal_img_smpls_train = random.sample(normal_img_files, 400)
covid_img_smpls_train = random.sample(covid_img_files, 400)


# In[5]:


normal_img_train = []
for img_name in normal_img_smpls_train:
    image = cv2.imread(normal_img_path + '/' + img_name, 1)
    image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    normal_img_train.append(np.array(image))
        
covid_img_train = []
for img_name in covid_img_smpls_train:
    image = cv2.imread(covid_img_path + '/' + img_name, 1)
    image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    covid_img_train.append(np.array(image))


# In[6]:


from matplotlib import pyplot as plt

covid_rndm = np.random.randint(0, len(covid_img_train))
rndm_covid_img = covid_img_train[covid_rndm]

normal_rndm = np.random.randint(0, len(normal_img_train))
rndm_normal_img = normal_img_train[normal_rndm]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
ax[0].imshow(np.uint8(rndm_covid_img))
ax[0].set_title('random covid CT')
ax[1].imshow(np.uint8(rndm_normal_img))
ax[1].set_title('random normal CT')
fig.show()


# In[7]:


#concatenate lists of normal and covid images
train_images = normal_img_train + covid_img_train

# 0: normal(non covid)  
# 1: covid
train_images_labels = np.concatenate((np.zeros(len(normal_img_train)), np.ones(len(covid_img_train))), axis=None, dtype=np.float32)

# free useless memory
normal_img_train = []
covid_img_train = []


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(np.array(train_images), np.array(train_images_labels), test_size=0.2, random_state=42)


# In[9]:


cnn_model = Sequential()
cnn_model.add(InputLayer(input_shape=(256,256,3)))
cnn_model.add(Lambda(lambda x: x/255.))   #Normalization

cnn_model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.1))
cnn_model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.1))
cnn_model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.3))
cnn_model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.3))
cnn_model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.4))
cnn_model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(Dropout(0.4))
cnn_model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Flatten())
cnn_model.add(Dense(4096, activation="relu"))
cnn_model.add(Dense(4096, activation="relu"))
cnn_model.add(Dense(1, activation="sigmoid"))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()


# In[ ]:


callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
             tf.keras.callbacks.ModelCheckpoint(filepath=r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\archive (2)\curated_data\curated_data\.ipynb_checkpoints', monitor='val_loss', mode='min')]

history = cnn_model.fit(x = np.asarray(X_train),
                        y = y_train,
                        batch_size = 20,
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
                        validation_data = (np.asarray(X_val), y_val),
                        verbose = 1,
                        epochs = 50)


# In[ ]:


val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
print('Best epoch = ', best_epoch)


# In[ ]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13,4))

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('model accuracy')
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'val'], loc='upper left')

ax[1].plot(history.history['loss'][1:])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('val_loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'val'], loc='upper right')

fig.show()


# In[ ]:


rest_of_normal_img_files = set(normal_img_files) - set(normal_img_smpls_train)
rest_of_covid_img_files = set(covid_img_files) - set(covid_img_smpls_train)


# In[ ]:


rest_of_normal_img_files = set(normal_img_files) - set(normal_img_smpls_train)
rest_of_covid_img_files = set(covid_img_files) - set(covid_img_smpls_train)

normal_img_smpls_test = random.sample(rest_of_normal_img_files, 200)
covid_img_smpls_test = random.sample(rest_of_covid_img_files, 200)


# In[ ]:


normal_img_test = []
for img_name in normal_img_smpls_test:
    image = cv2.imread(normal_img_path + '/' + img_name, 1)
    image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    normal_img_test.append(np.array(image))
        
covid_img_test = []
for img_name in covid_img_smpls_test:
    image = cv2.imread(covid_img_path + '/' + img_name, 1)
    image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    covid_img_test.append(np.array(image))


# In[ ]:


test_images = normal_img_test + covid_img_test
test_images_labels = np.concatenate((np.zeros(len(normal_img_test)), np.ones(len(normal_img_test))), axis=None, dtype=np.float32)

normal_img_test = []
covid_img_test = []


# In[ ]:


cnn_model.evaluate(x = np.asarray(test_images),
                   y = test_images_labels,
                   verbose=1)


# In[ ]:


from sklearn.metrics import classification_report

model_prd = cnn_model.predict(x = np.asarray(test_images))
for i in range(len(model_prd)):
    if model_prd[i] >= 0.5:
        model_prd[i] = 1
    elif model_prd[i] < 0.5:
        model_prd[i] = 0

classes = ['Class Normal', 'Class Covid']
print(classification_report(y_true = test_images_labels,
                      y_pred = model_prd,
                      target_names = classes))

