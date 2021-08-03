# 숫자 이미지 (MNIST)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# train/test 분류
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(len(x_train), len(y_train), len(x_test), len(y_test))
# print(x_train.shape, y_train.shape) # 60000개
# print(x_test.shape, y_test.shape)   # 10000개
# print(x_train[0])    인

# 3차원을 2차원으로 바꿔줌 (28X28 형태기 때문에 784)
# 정규화 작업을 위해 형태변환(float32으로)
x_train = x_train.reshape(60000, 784).astype('float32') 
x_test = x_test.reshape(10000, 784).astype('float32') 

# 학습을 위해서 0~255의 값을 0~1사이로 변환 해주기위해 최대값인 255로 나눠줘서 정규화
x_train /= 255
x_test /= 255
# print(x_train[0]) # 확인
    
# 그래프가 숫자형태로 나오는지 확인
# conda install matplotlib 
# import matplotlib.pyplot as plt 해줘야함
# plt.imshow(x_train[1].reshape(28, 28), cmap='Greys')    # 형태 28X28이어야 함 
# cmap은 colormap 색깔 설정으로 이해 (색의 종류는 https://jrc-park.tistory.com/155) 
# plt.show()

print(y_train[0])
print(set(y_train)) # y_train값의 unique값 (list타입을 set타입으로 변경해주면 set타입은 고유값만 가지는 특성이있어 동일한 값이 존재해도 하나만 나오는 특성을 활용)

# keras.utils.utils에 to_categorical은 one-hot인코딩을 해주는 함수(여기서는 0부터9까지 인코딩)
y_train = tf.keras.utils.to_categorical(y_train, 10)   # 괄호안에 처음은 인코딩해줄 값 뒤에 값은 인코딩해줄때 배열의 크기
y_test = tf.keras.utils.to_categorical(y_test, 10)
print(y_train[0]) # 확인 (y_train[0]은 5임)

# train dataset의 일부를 validation dataset
x_val = x_train[50000:] # x_val을 x_train데이터의 50000번째부터 60000번째까지 데이터로 설정하겠다는 의미 (만약 [:50000] 이면 0~50000 의 데이터 [50000:]이면 50000이후의 모든 데이터)
y_val = y_train[50000:60000]
x_train = x_train[0:50000]
y_train = y_train[:50000]
print(x_val.shape, '/', x_train.shape)

# 딥러닝 모델 구조 설정(2개층, 512개의 뉴런 연결, 10개 클래스 출력 뉴런, 784개 픽셀 input 값, relu와 softmax 활성화 함수 이용)
# 다층 퍼셉트론(MLP, Multilayer Perceptron)을 사용
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, input_shape = x_train.shape[1:], activation='relu'))
model.add(tf.keras.layers.Dropout(0.2)) # 위에서 설정한 512개중에서 0.2(20%)는 사용하지 않음 과적합 방지 // regularizers.l2(0.001)같은 방식으로도 과적합 방지가능(기억안나면 검색)
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 10이라는 숫자는 데이터의 크기?를 지정하는듯함 (0~9까지 존재하니 10개)


# 딥러닝 구조 설정(loss 옵션을 다중 클래스에 적합한 categorical_crossentropy, 옵티마이저는 adam 설정 - lr(learning rate)은 학습률)
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(lr=0.01),\
              metrics=['acc'])
print(model.summary())

###################################################################################
###################################################################################
# 학습모델을 저장해 한번만 훈련을 하게 만들어준다
# 훈련
# 학습횟수가 너무 많으면 과적합(overfitting)을 일으키고 너무 적은 Epoch은 underfitting(과소적합)을 일으킨다. 
# 성능이 더이상 증가하지 않으면 학습을 중지 시킴

from tensorflow.keras.callbacks import EarlyStopping
e_stop = EarlyStopping(patience=6, monitor='loss')    
# monitor값에 지정한 성능을 확인, patience는 성능이 증가하지 않다고 판단된 경우를 몇번까지 허용할 것인가

# 모델 실행(x_val, y_val로 검증, 256개씩  1000번 학습 - EarlyStop이 있기 때문에 많은 수치 입력)
history = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val),\
                     verbose=1, callbacks=[e_stop])
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
                     
# verbose = 학습 중 출력되는 문구를 설정합니다.(학습이 되는 모습을 볼수있다.)
# 0 : 아무 것도 출력하지 않습니다. 1 : 훈련의 진행도를 보여주는 진행 막대를 보여줍니다. 2 : 미니 배치마다 손실 정보를 출력합니다.

loss, acc = model.evaluate(x_test, y_test)
print(history.history.keys())
print('loss : ', history.history['loss'], ', val_loss : ', history.history['val_loss'])
print('acc : ', history.history['acc'], ', val_acc : ', history.history['val_acc'])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
score = model.evaluate(x_test, y_test)
print('score loss : ', score[0])
print('score acc : ', score[1])
model.save('mnist.hdf5')    # 데이터 저장(hdf5로) - https://aileen93.tistory.com/97

###################################################################################
###################################################################################
'''
model = tf.keras.models.load_model('mnist.hdf5')

# 예측 
pred = model.predict(x_test[:1])
print('예측값 : ', pred)

# import numpy as np
print('예측값 : ',np.argmax(pred,1))

print('실제값 : ', y_test[:1])
print('실제값 : ', np.argmax(y_test[:1], 1))

# 직접 데이터를 넣어 값 예측해보기
print('-----------------------------------')
# 이미지 처리를 위한 라이브러리 PIT호출
from PIL import Image
for i in range(10):
    im = Image.open('images/num'+str(i)+'.png')
    #img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert('L'))
    img = im.convert('L')   # L(256단계 흑백이미지)로 변환
    # print(img, img.shape)
    
    plt.imshow(img, cmap='Greys')
    plt.show()
    
    img = np.resize(img, (1, 784))
    data = ((np.array(img) / 255) - 1) * -1
    # print(data)
    
    new_pred = model.predict(data)
    print('new_pred : ', new_pred)
    print(np.argmax(new_pred, 1))
'''

# 참고 - https://saynot.tistory.com/entry/Deep-Learning-MNIST-%EC%86%90%EA%B8%80%EC%94%A8-%EC%88%AB%EC%9E%90-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9D%B8%EC%8B%9D%ED%95%B4%EB%B3%B4%EA%B8%B0
# 학습률 설정 참고 - https://forensics.tistory.com/28 