import matplotlib.pyplot as plt  # 그림으로 보기 위한 matplotlib 라이브러리 import
from tensorflow.keras.datasets import mnist  # 라이브러리가 기본으로 제공하는 mnist 데이터셋
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Flatten

# 과정 생각해보기
# 입력데이터 x(28 x 28), y(10) 를 불러온다.
# 입력데이터 x에 대한 가중치 배열을 x와 동일한 크기로 생성, 편향 b도 생성해준다.
# y의 값의 범위가 0 ~ 9 이기 때문에 소프트맥스 함수를 통해 값 얻어내기
# y의 값을 one hot encoding 하여 얻어내고, 결과값과 실제값을 비교

# 우선 x1w1 * x2w2 * x3w3 ... + b 가 되어야 겠지?

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# fan_in : input neuron
# fan_out : output neuron

def xavier_init(fan_in, fan_out, w):
    w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
    return w

def sigmoid(x):
    return 1.0 / (1.0 + np.e ** (-x)) # np.e = 지수 2.7xxx

def softmax(x_data):
    m = np.argmax(x_data)
    x_data = x_data - m
    x_data = np.exp(x_data)
    return x_data / np.sum(x_data)


X_train, X_test = X_train / 255.0 , X_test / 255.0 # 0 ~ 255 의 값을 0 ~ 1 사이로 변환

y_train_hot = np.eye(10)[y_train] # y_train 의 값을 10개의 리스트에서 표현

accurancy = 0

if __name__ == '__main__':
    for i in range(1000):
        w1_size = (2, 784)
        x1 = np.array(X_train[i]).flatten()
        w1 = np.zeros(784)
        w1 = xavier_init(784, 100, w1)
        b1 = np.zeros(100)
        # w1 = xavier_init(784, 2, w1)
        # w2 = np.zeros(shape=784)  # 가중치
        b2 = np.zeros(10)
        n1 = sigmoid(np.dot(x1, w1) + b1)
        # print(y_train_hot)

        w2 = np.zeros(100)
        w2 = xavier_init(100, 10, w2)
        n2 = np.dot(n1, w2) + b2

        result = softmax(n2)

        predict = np.argmax(result) == np.argmax(y_train_hot[i])
        # print(np.argmax(result))
        # print(predict)
        if predict == True:
            accurancy += 1.0

        if i % 100 == 0:
            print("accurancy : ", accurancy)



        # for x_data, y_data in data:
        #     a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
        #     b_diff = sigmoid(a * x_data + b) - y_data
        #     a = a - lr * a_diff
        #     b = b - lr * b_diff

    # plt.imshow(X_train[0])
    # plt.show()