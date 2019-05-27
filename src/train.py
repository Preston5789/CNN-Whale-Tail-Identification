
# coding: utf-8

# In[ ]:


from skimage import io
import pandas as pd
import numpy as np
import tensorflow as tf
import os


# In[ ]:


x_label = pd.read_csv('train.csv')
TRAIN_DATA_PATH = '../train/'
IMAGES = []
IMAGE_LIST = x_label.Image.values[x_label.Id.values != 'new_whale']

for img in IMAGE_LIST:
    IMAGES.append(io.imread(TRAIN_DATA_PATH + img))
IMAGES = np.array(IMAGES).astype('float32')/255


# In[ ]:


x_train = IMAGES.copy()
x_train = x_train.reshape(9040,100,100,1)
y_train = x_label.Id.values[x_label.Id.values != 'new_whale']


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = y_train
values = np.array(data)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train_onehot = onehot_encoder.fit_transform(integer_encoded)
y_train_onehot.shape


# In[ ]:


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint


# In[ ]:


model = Sequential()
    
model.add(Conv2D(input_shape=(100,100,1),filters = 64, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 64, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 256, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size=(3,3), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4250, activation='softmax'))

from keras.optimizers import SGD,Adam, RMSprop
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00123),metrics=['accuracy'])
model.summary()


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=8,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(x_train)


# In[ ]:


import os
import datetime
curr = str(datetime.datetime.now())
curr = curr.split(' ')
model_dir = '../model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
save_path = 'model/%s_%s_model.h5'%("".join(curr[0].split('-')[1:]),"".join(curr[1].split(":")[0:2]))
es = EarlyStopping(monitor='acc', patience=10, verbose=1, mode='max')
cp = ModelCheckpoint(filepath=save_path,verbose=1,save_best_only=True,save_weights_only=False,monitor='acc',mode='max')


# In[ ]:


train_history = model.fit_generator(datagen.flow(x_train,  
                                                 y_train_onehot, batch_size=16),
                                    steps_per_epoch=round(len(x_train)/16),
                                    epochs=400,
                                    validation_data=(x_train, y_train_onehot),
                                    callbacks=[cp,es])

