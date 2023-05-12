#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
import time
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# In[15]:


BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = 175
BASE_DIR = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\archive (2)\chest_xray\chest_xray'
LABELS = ['NORMAL', 'PNEUMONIA']

train_dir = os.path.join(BASE_DIR, 'train')
val_dir = os.path.join(BASE_DIR, 'val')
test_dir = os.path.join(BASE_DIR, 'test')


# In[16]:


def get_preview_data(data_dir):
    data = []
    for label in LABELS:
        path = os.path.join(data_dir, label)
        class_num = LABELS.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([img_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# In[17]:


val = get_preview_data(val_dir)
train = get_preview_data(train_dir)
test = get_preview_data(test_dir)


# In[18]:


x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
    
# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# Resize for the model
x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)


# In[19]:


fig, ax = plt.subplots(1, 4, figsize=(12, 12))
ax = ax.flatten()
ax[0].imshow(val[0][0], cmap='gray')
ax[0].set_title(LABELS[val[0][1]])
ax[1].imshow(val[14][0], cmap='gray')
ax[1].set_title(LABELS[val[14][1]])
ax[2].imshow(val[6][0], cmap='gray')
ax[2].set_title(LABELS[val[6][1]])
ax[3].imshow(val[11][0], cmap='gray')
ax[3].set_title(LABELS[val[11][1]])
plt.tight_layout()
plt.show()


# In[20]:


img_gen_train = ImageDataGenerator(
    horizontal_flip = True,
    zoom_range = 0.2,
    rotation_range = 30,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    fill_mode = 'nearest'
)

img_gen_def = ImageDataGenerator()


# In[21]:


img_gen_train.fit(x_train)


# In[22]:


def plot_augmented_img(img_arr):
    fig, axs = plt.subplots(1, 5, figsize=(20, 20))
    axs = axs.flatten()
    for img, ax in zip(img_arr, axs):
        ax.imshow(np.squeeze(img), cmap='gray')
    plt.tight_layout()
    plt.show()


# In[23]:


train_aug_imgs = [x_train[i] for i in range(5)]
plot_augmented_img(train_aug_imgs)


# In[24]:


callback = []
callback.append(ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001))
callback.append(ModelCheckpoint('model-checkpoint.h5', monitor='val_loss', save_best_only=True))
callback.append(EarlyStopping(patience=25, monitor='val_loss'))


# In[25]:


model = Sequential([
    Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=2, padding='same'),
    
    Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=2, padding='same'),
    
    Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=2, padding='same'),
    
    Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=2, padding='same'),
    
    Conv2D(256, (2, 2), strides=1, padding='same', activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=2, padding='same'),
    
    Flatten(),
    Dense(units=256, activation='relu'),
    Dropout(0.3),
    Dense(units=128, activation='relu'),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
])

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# In[26]:


history = model.fit(
    img_gen_train.flow(x_train, y_train, batch_size=32),
    epochs=EPOCHS,
    steps_per_epoch=163,
    validation_data=img_gen_def.flow(x_val, y_val),
    callbacks=[callback]
)

t = time.time()
export_path_keras = "./pneumonia-model2-{}.h5".format(int(t))
model.save(export_path_keras)


# In[1]:


model = keras.models.load_model('./model-checkpoint.h5')


# In[31]:


epoch_range = range(33)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[32]:


fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(epoch_range, acc, label='Training Accuracy')
ax[0].plot(epoch_range, val_acc, label='Validation Accuracy')
ax[0].set_xlabel('Training Epoch')
ax[0].set_ylabel('Accuracy (%)')
ax[0].set_title('Model Accuracy Variation over Training Period')
ax[0].legend()

ax[1].plot(epoch_range, loss, label='Training Loss')
ax[1].plot(epoch_range, val_loss, label='Validation Loss')
ax[1].set_xlabel('Training Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_title('Model Loss Variation over Training Period')
ax[1].legend()

plt.grid(True)
plt.show()


# In[33]:


print('Model Accuracy:', model.evaluate(img_gen_def.flow(x_test, y_test, batch_size=32))[1])
print('Model Loss:', model.evaluate(img_gen_def.flow(x_test, y_test, batch_size=32))[0])

