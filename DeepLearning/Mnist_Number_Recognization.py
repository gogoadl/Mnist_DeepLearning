import matplotlib.pyplot as plt  # 그림으로 보기 위한 matplotlib 라이브러리 import
from tensorflow.keras.datasets import mnist  # 라이브러리가 기본으로 제공하는 mnist 데이터셋
import numpy as np
from keras.datasets import mnist

# fan_in : input neuron
# fan_out : output neuron

def xavier_init(fan_in, fan_out):
    w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in) # Xavier 초기화 방법은 표준 정규 분포를 입력 개수의 표준 편차로 나누는 방법이다.
    return w

def sigmoid(x):
    return 1.0 / (1.0 + np.e ** (-x)) # np.e = 지수 2.7xxx

def softmax(x_data):
    m = np.argmax(x_data)
    x_data = x_data - m
    x_data = np.exp(x_data)
    return x_data / np.sum(x_data)

(X_train, y_train), (X_test, y_test) = mnist.load_data() # mnist에서 데이터셋을 가져온다.

X_train, X_test = X_train / 255.0 , X_test / 255.0 # 0 ~ 255 의 값을 0 ~ 1 사이로 변환

y_train_hot = np.eye(10)[y_train] # y_train 의 값을 10개의 리스트에서 표현

accurancy = 0

epochs = 2000

if __name__ == '__main__':
    max_arr = []
    min_arr = []
    for i in range(epochs):

        input_x = np.array(X_train[i]).flatten() # 28 X 28 배열을 일차원 배열으로 바꿔주기
        input_w = xavier_init(784, 2) # xavier 가중치 초기화
        input_b = np.zeros(2) # bias 초기화

        hidden_w1 = xavier_init(2, 3)
        hidden_b1 = np.zeros(3)

        print("------")
        # print("hidden_w1", hidden_w1)
        # print("hidden_w1 최댓값", np.max(hidden_w1))
        # print("hidden_w1 최솟값", np.min(hidden_w1))
        max_arr.append(np.max(hidden_w1))
        min_arr.append(np.min(hidden_w1))
        print("------")
        hidden_w2 = xavier_init(3, 10)
        hidden_b2 = np.zeros(10)

        hidden_n1 = sigmoid(np.dot(input_x, input_w) + input_b)

        hidden_n2 = sigmoid(np.dot(hidden_n1, hidden_w1) + hidden_b1)
        # print(hidden_n2)
        output_n = np.dot(hidden_n2, hidden_w2) + hidden_b2

        result = softmax(output_n)
        # print(result)
        predict = np.argmax(result) == np.argmax(y_train_hot[i])
        # print(np.argmax(result))
        # print(predict)
        if predict:
            accurancy += 1.0

    accurancy = accurancy / epochs

    print("accuracy : %.4f" % (accurancy))
    print("max_arr 최댓값", np.max(max_arr))
    print("min_arr 최솟값", np.min(min_arr))
    # plt.imshow(X_train[0])
    # plt.show()