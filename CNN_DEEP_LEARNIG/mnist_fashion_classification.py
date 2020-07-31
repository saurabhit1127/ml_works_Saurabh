# -*- coding: utf-8 -*-
"""mnist_fashion_classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lPAMm3C5JZjFrMK-lzcdSgyvyq7gcKqR
"""

import keras 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

"""load data from keras"""

(X_train, y_train), (X_test, y_test)= keras.datasets.fashion_mnist.load_data()

X_train.shape,y_train.shape

X_test.shape,y_test.shape

X_train[0] # one image

y_train[0] # 0th item is in 9 th class

class_label=["t-shirt","trouser","pullover","Dress","coat","sandal","shirt","sneaker","bag","ankle boot"]

plt.imshow(x_train[0],cmap='Greys')  #bad image because of less pixel ,generally we decrease the resolution of  image (downsample),helps in less computation of neural n/w

plt.figure(figsize=(16,16))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(X_train[i])
  plt.axis('off')
  plt.title(class_label[y_train[i]]+"="+str(y_train[i]),fontsize=16)

"""feature scaling"""

X_train = X_train/255
X_test = X_test/255

X_train[0]

"""neural network train

flatten means we are converting the data  in vectors,here we are giving the input into the flatten. creating first neural layer (called dense layer and how many neurons you want to take(units) and give the activation function.In the output layer actually you have to take the no of neuron (units)= no of classes and for each of the classes, we are  finding the probability for each of the class therfore the activation function we have to take is softmax function.
"""

model= keras.models.Sequential([keras.layers.Flatten(input_shape=[28,28]),
                         keras.layers.Dense(units=32,activation='relu'),
                         keras.layers.Dense(units=10,activation='softmax')

                         
                         
                         ])

model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

"""epoch means suppose we have 60k images will be pass through neural network in one go, to train the network well increase the epocs,"""

model.fit(X_train,y_train,epochs=10)

"""test and evaluate neural network"""

model.evaluate(X_test,y_test)

Y_pres=model.predict(X_test)

"""In the 9th index the probability is high that means the model predicts that the first image is belong to the 9th class ."""

Y_pres[0].round(2)

"""if you want find the index of the element use argmax. it finds you the index which have the the high probabilty"""

index = np.argmax(Y_pres[0].round(2))
print(index)

y_test[0] # checked with test data

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,[np.argmax(i) for i in  Y_pres])

plt.figure(figsize=(16,9))
sns.heatmap(cm,annot=True,fmt = 'd')



from sklearn.metrics import classification_report
cr=classification_report(y_test,[np.argmax(i) for i in  Y_pres],target_names = class_label,)
print(cr)

