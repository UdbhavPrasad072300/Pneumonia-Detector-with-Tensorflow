#!/usr/bin/env python
# coding: utf-8

# # Detecting Pneumonia through X-ray Images with Convolutional Neural Networks in Keras and Back-end Tensorflow

# Written by: Udbhav Prasad <br>
# Linkedin: https://www.linkedin.com/in/udbhav-prasad-1506b7192/ <br>
# HackerRank: https://www.hackerrank.com/uprasad1 <br>
# Github: https://github.com/UdbhavPrasad072300 <br>
# Computer Science Co-op - Ryerson University <br> <hr>
# Making a Convolutional Neural Network with Keras and back-end Tensorflow to detect whether someone has pneumonia based on an X-ray scan of their chest <br>
# An accuracy of <b>96%</b> was achieved for Training Set & <br>
# An accuracy of <b>90%</b> was achieved for Validation Set 

# ## Importing Modules

# In[1]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


# ## Checking if GPU is being used by Tensorflow

# My CPU: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz, 3696 Mhz, 6 Core(s), 12 Logical Processor(s) <br>
# My GPU: NVIDIA GeForce GTX 1080 8GB Dedicated Memory

# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ## Initialising the Convolutional Neural Network

# In[3]:


#strategy = tf.distribute.MirroredStrategy(devices=None, cross_device_ops=None)

#with strategy.scope():
classifier = Sequential()


# ## Adding Layers to Neural Network

# First Convolutional Layer

# In[4]:


classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) # Step 1 - Convolution
classifier.add(MaxPooling2D(pool_size = (2, 2))) # Step 2 - Pooling


# Second Convolutional Layer

# In[5]:


classifier.add(Conv2D(32, (3, 3), activation = 'relu')) # Step 1 - Convolution
classifier.add(MaxPooling2D(pool_size = (2, 2))) # Step 2 - Pooling


# Flattening Layer <br>
# Turns into 1 dimension for input layer

# In[6]:


classifier.add(Flatten())


# Arificial Neural Network

# In[7]:


classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# ## Compiling Convolutional Neural Network 

# In[8]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ## Fitting Images to Convolutional Neural Network 

# In[9]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[10]:


training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# ## Summary of the Layers and Characteristics of the Convolutional Neural Network

# In[11]:


classifier.summary()


# ## Training the Network with Data

# Steps per epoch: 250 <br>
# Num. of epochs: 15 <br>

# In[12]:


history = classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 200)


# Just checking the GPU

# In[13]:


tf.test.gpu_device_name()


# Returns a dictionary of the Configuration of the Classifier

# In[14]:


classifier.get_config()


# If you wanna see some of the trained weights of the Classifier

# In[15]:


#from itertools import islice
#print(*islice(classifier.get_weights(), 1))


# ## Saving the Classifier for Later Use

# In[16]:


classifier.save("pneumoniaDetector.h5")
print("Saved model to disk")


# ## Loading in the Model just to confirm that it actually works

# In[17]:


import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model("pneumoniaDetector.h5")
test_image = image.load_img("person2_bacteria_4.jpeg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = "OMG YOU HAVE PNEUMONIA" 
else:
    prediction = "you good man, dw"
print(prediction) # Expected: "OMG YOU HAVE PNEUMONIA"


# ## Analyzing the History of the Trained Convolutional Neural Network [Accuracy & Validation Accuracy]

# Validation Test Set Accuracy and the Training Set

# In[18]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'])
plt.show()


# ## Conclusion

# Test Set ended with an Accuracy of 97.27% on Training Set and 91.03% on Validation Set

# ## Improvements that can be made

# Adding more convolutional layers would have helped because images were very high resolution

# And obviously training with more epochs 
