# 필요한 라이브러리 불러오기 
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt

# if __name__ == '__main__':
#     N = 10000 ### 무작위시행 횟수 정의
#     x = np.random.random([N, 2])    # N번 실행 / 2는 2차원, N개의 점이 각각 (x, y) 값을 가짐
#     distance = np.sum(x ** 2.0, axis=1) # (x, y)에서 distance = x²+y² / axis=1은 행을 더한다는 뜻
#     in_out = distance <= 1.0
#     pi = np.sum(in_out)*4/N ### Pi 값은 천제 시행에서 원 안에 있는 점의 갯수로 정해짐
#     color = list(map(lambda x: 'red' if x else 'blue', in_out)) ### 원의 안, 밖에 따른 색상 설정

#     plt.figure(figsize=(5, 5)) ### 그림 사이즈
#     plt.scatter(x[:,0], x[:,1], color = color, s=5, label ='Result : {}'.format(np.round(pi, 4)))
    
#     cx = np.cos(np.linspace(0, np.pi/2, 1000))
#     cy = np.sin(np.linspace(0, np.pi/2, 1000))
#     plt.plot(cx, cy, color = 'black', lw =2) ### 원의 경계를 그려주는 부분
#     plt.legend(loc = 'lower right')

#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.show()

# 2. 예시 2: 몬테카를로 - 원주율 추정
n_points = 100000  # 난수 점 개수

x_points = np.random.uniform(-1, 1, n_points)  # x 좌표
y_points = np.random.uniform(-1, 1, n_points)  # y 좌표

# 원 내부 점 계산
distances = x_points**2 + y_points**2   # 리스트 형식
inside_circle = distances <= 1          # bool형 리스트

print(distances, inside_circle)

# 원주율 계산
pi_estimate = 4 * np.sum(inside_circle) / n_points

print(np.sum(inside_circle) / n_points)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(x_points[inside_circle], y_points[inside_circle], color="blue", s=1, label="Inside Circle")
plt.scatter(x_points[~inside_circle], y_points[~inside_circle], color="red", s=1, label="Outside Circle")
plt.gca().set_aspect('equal')
plt.title(f"Example 2: Monte Carlo - Estimating π (π ≈ {pi_estimate:.4f})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
