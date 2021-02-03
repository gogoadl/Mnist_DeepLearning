import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부 시간 X와 성적 Y의 리스트 만들기
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data] # 2, 4, 6, 8
y = [i[1] for i in data] # 81, 93, 91, 97

# 그래프로 나타내기
plt.figure(figsize=(8,5))
plt.scatter(x, y)
plt.show()

# 리스트로 되어 있는 x와 y 값을 넘파이 배열로 바꾸기
x_data = np.array(x)
y_data = np.array(y)

#기울기 a와 절편 b의 값 초기화
a = 0
b = 0

# 학습률 정하기
lr = 0.03

# 몇번 반복될지 설정
epochs = 2001

# 경사 하강법 시작
for i in range(epochs):
    y_pred = a * x_data + b
    error = y_data - y_pred
    # 오차 함수를 a로 미분한 값
    a_diff = -(2/len(x_data)) * sum(x_data * (error))
    # 오차 함수를 b로 미분한 값
    b_diff = -(2 / len(x_data)) * sum(error)

    a = a - lr * a_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print("epoch=%.f 기울기=%.04f 절편=%.04f" % (i, a, b))
        print(a_diff, b_diff)

# 앞서 구한 기울기와 절편을 이용해 그래프를 다시 그리기
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()