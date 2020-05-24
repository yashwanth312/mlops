#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Dense


# In[4]:


from keras.layers import Flatten


# In[5]:


from keras.models import Sequential


# In[6]:


model=Sequential()


# In[7]:


model.add(Convolution2D(filters=64 , kernel_size=(3,3) , activation='relu' , input_shape=(64,64,3)))


# In[8]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[9]:


model.add(Convolution2D(filters=32 , kernel_size=(3,3) , activation='relu'))


# In[10]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[12]:


model.add(Flatten())


# In[13]:


model.add(Dense(units=512 , activation='relu'))


# In[14]:


model.add(Dense(units=128 , activation='relu'))


# In[15]:


model.add(Dense(units=64 , activation='relu'))


# In[16]:


model.add(Dense(units=32 , activation='relu'))


# In[17]:


model.summary()


# In[18]:


from keras.optimizers import Adam


# In[19]:


model.compile(optimizer='Adam' , loss='binary_crossentropy' , metrics=['accuracy'])


# In[20]:


from keras_preprocessing.image import ImageDataGenerator


# In[21]:


from keras.preprocessing import image


# In[22]:


model.add(Dense(units=1 , activation='sigmoid'))


# In[24]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=1,
        validation_data=test_set,
        validation_steps=800)


# In[ ]:




