import numpy as np
import matplotlib.pyplot as plt

# 1s 오비탈의 확률 밀도 함수 정의
def hydrogen_1s_probability_density(r):
    """
    1s 오비탈의 정규화된 확률 밀도 함수.
    Psi(r)^2 = (1/π)*(1/a0^3)*exp(-2r/a0)로 정의되며, 여기서 a0는 보어 반지름입니다.
    이 구현에서는 보어 반지름(a0)을 1로 가정하고 원자 단위를 사용합니다.
    
    Parameters:
        r (float): 전자가 핵에서 떨어진 거리 (반지름).
    
    Returns:
        float: 주어진 r에서의 확률 밀도 값.
    """
    a0 = 1  # 보어 반지름을 1로 가정 (원자 단위)
    return (np.exp(-2 * r) / np.pi)  # 정규화된 확률 밀도 값 반환

# 구면 좌표를 고려한 확률 밀도 함수 정의
def hydrogen_1s_probability_density_spherical(r):
    """
    구면 좌표에서의 확률 밀도 함수. 
    구면 대칭을 고려하여 4πr^2 요소를 포함합니다.
    이를 통해 r이 증가할 때 구의 부피에 따라 확률이 조정됩니다.
    
    Parameters:
        r (float): 전자가 핵에서 떨어진 거리 (반지름).
    
    Returns:
        float: 구면 좌표에서의 확률 밀도 값.
    """
    return 4 * np.pi * r**2 * hydrogen_1s_probability_density(r)

# Metropolis-Hastings 알고리즘을 사용한 MCMC 샘플링
def mcmc_metropolis_hastings_spherical(num_samples, step_size, burn_in=1000):
    """
    Metropolis-Hastings 알고리즘을 사용하여 1s 오비탈에서 샘플을 생성.
    구면 좌표계의 확률 밀도를 기반으로 샘플을 생성합니다.
    
    Parameters:
        num_samples (int): MCMC에서 생성할 최종 샘플 개수.
        step_size (float): 샘플 제안 분포의 스텝 크기 (탐색 범위).
        burn_in (int): 초기 샘플을 버리는 단계의 반복 횟수 (버닝 단계).
    
    Returns:
        numpy.ndarray: 생성된 샘플의 배열.
    """
    samples = []  # 최종 샘플을 저장할 리스트
    current_position = np.random.rand()  # 초기 위치를 랜덤하게 설정
    current_prob = hydrogen_1s_probability_density_spherical(current_position)  # 초기 위치의 확률 계산

    for i in range(num_samples + burn_in):  # 버닝 단계 포함하여 반복
        # 대칭적인 제안 분포에서 새로운 위치 제안
        proposal = current_position + np.random.uniform(-step_size, step_size)
        if proposal < 0:  # r은 음수가 될 수 없으므로 무시
            continue

        # 제안된 위치에서의 확률 밀도 계산
        proposal_prob = hydrogen_1s_probability_density_spherical(proposal)

        # 수락 비율 계산 (제안된 위치의 확률 / 현재 위치의 확률)
        acceptance_ratio = proposal_prob / current_prob

        # 제안된 위치를 수락 또는 거절
        if np.random.rand() < acceptance_ratio:
            current_position = proposal  # 새로운 위치로 이동
            current_prob = proposal_prob  # 새로운 위치의 확률 업데이트

        # 버닝 단계를 지나고 나면 샘플 저장
        if i >= burn_in:
            samples.append(current_position)

    return np.array(samples)  # 배열로 변환하여 반환

# MCMC 매개변수 설정
num_samples = 200000  # 최종 생성할 샘플 개수
step_size = 0.5  # 샘플 제안 분포의 스텝 크기
burn_in = 2000  # 버닝 단계 반복 횟수

# MCMC를 사용하여 샘플 생성
samples_spherical = mcmc_metropolis_hastings_spherical(num_samples, step_size, burn_in)

# 결과 시각화
plt.figure(figsize=(8, 6))  # 그래프 크기 설정
plt.hist(samples_spherical, bins=100, density=True, alpha=0.7, color='blue', label='MCMC 샘플 (구면 조정)')  # 히스토그램
r = np.linspace(0, 5, 500)  # r 값의 범위 설정
plt.plot(r, hydrogen_1s_probability_density_spherical(r), 'r-', label='이론적 분포 (구면)', linewidth=2)  # 이론적 분포
plt.xlabel('Radius (r)')  # x축 라벨
plt.ylabel('Probability Density')  # y축 라벨
plt.title('1s 오비탈 확률 분포 (구면 MCMC)')  # 그래프 제목
plt.legend()  # 범례 추가
plt.grid()  # 그리드 추가
plt.show()  # 그래프 표시
