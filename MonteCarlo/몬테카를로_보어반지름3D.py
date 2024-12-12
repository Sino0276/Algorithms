import numpy as np
import matplotlib.pyplot as plt

# 상수 정의
a0 = 1.0  # 보어 반지름, 전자가 존재하는 평균 거리를 나타내며 임의 단위로 설정

# 확률 밀도 함수 정의
def probability_density(r):
    """
    1s 오비탈의 확률 밀도 함수.
    주어진 거리 r에서의 확률 밀도 값을 계산합니다.

    Args:
        r (float): 반지름(전자와 원자핵 사이의 거리)

    Returns:
        float: 해당 거리 r에서의 확률 밀도
    """

    waveFunc_Square = (1 / (np.pi * a0**3)) * np.exp(-2 * r / a0) # 파동함수의 제곱

    # *구면 좌표*에서 부피 요소(4πr^2)를 포함한 확률 밀도
    return 4 * np.pi * r**2 * waveFunc_Square

# 샘플링에 사용할 변수 초기화
num_samples = 100000  # 시도할 총 샘플 수
max_r = 5 * a0  # 반지름 r의 최대값. 전자가 존재할 수 있는 최대 거리를 설정
accepted = []  # 수락된 샘플의 좌표를 저장할 리스트

# 샘플 생성 및 수락/거부 과정
for _ in range(num_samples):
    # (1) 무작위로 r, θ, φ 값을 생성
    # r: 반지름, [0, max_r] 범위에서 균일 분포로 난수 생성
    r = np.random.uniform(0, max_r)
    # θ: 방위각, [0, π] 범위에서 균일 분포로 난수 생성
    theta = np.random.uniform(0, np.pi)
    # φ: 방위각, [0, 2π] 범위에서 균일 분포로 난수 생성
    phi = np.random.uniform(0, 2 * np.pi)
    
    # (2) 확률 밀도의 최대값을 기준으로 수락/거부 결정
    r_max = a0 / 2
    max_prob = probability_density(r_max)  # 확률 밀도의 이론적 최대값 (r = a0 / 2)
    # print(max_prob)
    # print(probability_density(r))
    # 0.36xx < probability
    if np.random.uniform(0, max_prob) < probability_density(r):
        # (3) 수락된 샘플을 데카르트 좌표로 변환
        # 구면 좌표를 데카르트 좌표로 변환하여 저장
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        accepted.append([x, y, z])  # 수락된 샘플을 리스트에 추가

# (4) 수락된 샘플 데이터를 NumPy 배열로 변환
accepted = np.array(accepted)

# (5) 결과를 3D 산점도로 시각화
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# 산점도로 각 좌표를 점으로 표시
ax.scatter(accepted[:, 0], accepted[:, 1], accepted[:, 2], s=1, alpha=0.5)
ax.set_title("1s 오비탈 전자 분포")  # 제목 설정
ax.set_xlabel("X (a.u.)")  # X축 레이블
ax.set_ylabel("Y (a.u.)")  # Y축 레이블
ax.set_zlabel("Z (a.u.)")  # Z축 레이블
plt.show()  # 그래프 출력
