import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# 공부 시간 X와 합격 여부 Y의 리스트 만들기
data = [[25, 0], [31, 0], [35, 0], [42, 1], [51, 1], [63, 1]]

x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

# 그래프로 나타내기
plt.scatter(x_data, y_data)
plt.xlim(0, 70)
plt.ylim(-.1, 1.1)

# 기울기 a와 y 절편 b 초기화
a = -10
b = 0

# 학습률
lr = 0.05

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.e ** (-x)) # np.e = 지수 2.7xxx


plt.scatter(x_data, y_data)
plt.xlim(0, 70)
plt.ylim(-.1, 1.1)
x_range = (np.arange(0, 70, 0.1))

# 경사 하강법 실행
if __name__ == '__main__':
    for i in range(2001):
        for x_data, y_data in data:
            a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
            b_diff = sigmoid(a * x_data + b) - y_data
            a = a - lr * a_diff
            b = b - lr * b_diff
        if i % 50 == 0:
            print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))
            plt.plot(np.arange(0, 70, 0.1), np.array([sigmoid(a * x + b) for x in x_range]))
            time.sleep(0.1)
            plt.pause(0.01)

    plt.show()