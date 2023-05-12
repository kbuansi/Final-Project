#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
import numpy as np

# Define the deep multi-modal CNN architecture
def create_multi_modal_cnn(input_shape_xray, input_shape_ct, num_classes):
    # Input layers for X-ray and CT images
    input_xray = Input(shape=input_shape_xray, name='input_xray')
    input_ct = Input(shape=input_shape_ct, name='input_ct')

    # X-ray branch
    xray_conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_xray)
    xray_pool1 = MaxPooling2D(pool_size=(2, 2))(xray_conv1)
    xray_conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(xray_pool1)
    xray_pool2 = MaxPooling2D(pool_size=(2, 2))(xray_conv2)
    xray_flatten = Flatten()(xray_pool2)

    # CT branch
    ct_conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_ct)
    ct_pool1 = MaxPooling2D(pool_size=(2, 2))(ct_conv1)
    ct_conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(ct_pool1)
    ct_pool2 = MaxPooling2D(pool_size=(2, 2))(ct_conv2)
    ct_flatten = Flatten()(ct_pool2)

    # Concatenate the X-ray and CT branches
    merged = Concatenate()([xray_flatten, ct_flatten])

    # Dense layers for classification
    dense1 = Dense(128, activation='relu')(merged)
    dense2 = Dense(num_classes, activation='sigmoid')(dense1)

    # Create the model
    model = Model(inputs=[input_xray, input_ct], outputs=dense2)

    return model

# Define input shapes and number of classes
input_shape_xray = (256, 256, 3)  # Example shape for X-ray images
input_shape_ct = (256, 256, 3)  # Example shape for CT images
num_classes = 2  # Example: COVID-positive and COVID-negative

# Create the multi-modal CNN model
model = create_multi_modal_cnn(input_shape_xray, input_shape_ct, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()


# In[8]:


# Define your training and validation data
xray_train = np.random.random((1000, 256, 256, 3))
ct_train = np.random.random((1000, 256, 256, 3))
labels_train = np.random.randint(0, 2, (1000, num_classes))

xray_val = np.random.random((200, 256, 256, 3))
ct_val = np.random.random((200, 256, 256, 3))
labels_val = np.random.randint(0, 2, (200, num_classes))



# In[9]:


# Define your training and validation data
xray_train = np.random.random((1000, 256, 256, 3))
ct_train = np.random.random((1000, 256, 256, 3))
labels_train = np.random.randint(0, 2, (1000, num_classes))

xray_val = np.random.random((200, 256, 256, 3))
ct_val = np.random.random((200, 256, 256, 3))
labels_val = np.random.randint(0, 2, (200, num_classes))



# In[10]:


# Define the number of epochs
epochs = 50

# Train the model
history = model.fit(
    [xray_train, ct_train],
    labels_train,
    validation_data=([xray_val, ct_val], labels_val),
    epochs=epochs,
    verbose=1
)


# In[11]:


# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:




