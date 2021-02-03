import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부 시간 X와 성적 Y의 리스트 만들기
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 그래프로 확인

ax = plt.axes(projection='3d')
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()

# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기 (인덱스로 하나씩 불러와 계산할 수 있도록 하기 위함)
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

a1 = 0
a2 = 0
b = 0

# 학습률
lr = 0.05

epochs = 2001

# 경사 하강법 시작

for i in range(epochs):
    y_pred = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_pred
    # 오차 함수를 a1로 미분한 값
    a1_diff = -(1 / len(x1_data)) * sum(x1_data * (error))
    # 오차 함수를 a2로 미분한 값
    a2_diff = -(1 / len(x2_data)) * sum(x2_data * (error))
    # 오차 함수를 b로 미분한 값
    b_diff = -(1 / len(x1_data)) * sum(y_data - y_pred)

    a1 = a1 - lr * a1_diff
    a2 = a2 - lr * a2_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print("epoch=%.f, 기울기1=%.04f 기울기2=%.04f, 절편=%.04f" % (i, a1, a2, b))


