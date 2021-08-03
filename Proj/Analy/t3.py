# MNIST로 CNN 연습
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)

# CNN : 3차원을 4차원을 구조 변경
train_images = train_images.reshape((60000, 28, 28, 1))
print(train_images.shape, train_images.ndim)
train_images = train_images / 255.0
# print(train_images[0])
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images / 255.0
print(train_labels[:3])

# 모델
'''
input_shape = (28, 28, 1)
model = models.Sequential()
# 형식 :  tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid', .....)
model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),\
                         padding='valid', activation='relu', input_shape = input_shape))   # padding='same'
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten()) # Fully Connected layer - CNN 처리된 데이터를 1차원 자료로 변경
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(10, activation = 'relu'))
print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss')
history = model.fit(train_images, train_labels, batch_size = 128, epochs=100, verbose=1,\
                    validation_split=0.2, callbacks=[early_stop])
train_loss, train_acc = model.evaluate(train_images, train_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)
'''

# 모델 저장
# model.save('ke21.h5')


import pickle
# history = history.history
# with open('data.pickle', 'wb') as f:
#     pickle.dump(history, f)

with open('data.pickle', 'rb') as f:
    history = pickle.load(f)
    
# django에 적용시킬때 미리 학습을 해놓은 데이터를 저장해 불러와 분석만 하면된다. 
model = tf.keras.models.load_model('ke21.h5')

# train_loss, train_acc = model.evaluate(train_images, train_labels)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('train_loss : ', train_loss)
# print('train_acc : ', train_acc)
# print('test_loss : ', test_loss)
# print('test_acc : ', test_acc)

# 예측
import numpy as np
print('예측값 : ', np.argmax(model.predict(test_images[:1])))
print('예측값 : ', np.argmax(model.predict(test_images[[0]])))
print('실제값 : ', test_labels[0])

print('예측값 : ', np.argmax(model.predict(test_images[[1]])))
print('실제값 : ', test_labels[1])

# acc와 loss로 시각화
import matplotlib.pyplot as plt

def plot_acc(title = None):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if title is not None:
        plt.title(title)
    plt.ylabel(title)
    plt.xlabel('epoch')
    plt.legend(['train data', 'validation data'], loc = 0)
    
plot_acc('accuracy')
plt.show()
    

def plot_loss(title = None):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel(title)
    plt.xlabel('epoch')
    plt.legend(['train data', 'validation data'], loc = 0)
    
plot_loss('loss')
plt.show()