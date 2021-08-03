# 이미지 분류 패션 MNIST
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_image, train_labels), (test_image, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot']
print(train_image.shape, train_labels.shape)    # (60000, 28, 28) (60000,)
print(set(train_labels))

# plt.imshow(train_image[0])
# plt.colorbar()
# plt.show()

'''
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i]])
    plt.imshow(train_image[i])
    
plt.show()
'''

# print(train_image[0])
train_image = train_image / 255.0
# print(train_image[0])
test_image = test_image / 255.0

# 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(512, activation = tf.nn.relu),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax),
    
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 한번 저장했기 때문에 더이상 학습 X
# model.fit(train_image, train_labels, batch_size=128, epochs=5, verbose=1)
# model.save('ke20.hdf5')

model = tf.keras.models.load_model('ke20.hdf5')

test_loss, test_acc = model.evaluate(test_image, test_labels)
print('loss : ', test_loss)
print('acc : ', test_acc)

pred = model.predict(test_image)
print(pred[0])
print('예측값  :', np.argmax(pred[0]))
print('실제값  :', test_labels[0])

# 각 이미지 출력용 함수
def plot_image(i, pred_arr, true_label, img):
    pred_arr, true_label, img = pred_arr[i], true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')
    
    pred_label = np.argmax(pred_arr)
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[pred_label], 100 * np.max(pred_arr), \
                                         class_names[true_label]), color = color)
    
i = 0
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, pred, test_labels, test_image)
plt.show()

def plot_value_arr(i, pred_arr, true_label):
    pred_arr, true_label = pred_arr[i], true_label[i]
    thisplot = plt.bar(range(10), pred_arr)
    plt.ylim([0, 1])

i = 12
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, pred, test_labels, test_image)
plt.subplot(1, 2, 2)
plot_image(i, pred, test_labels)
plt.show()