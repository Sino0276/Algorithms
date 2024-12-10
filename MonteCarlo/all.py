# Re-import necessary libraries after reset
import numpy as np
import matplotlib.pyplot as plt

# 1. 예시 1: 마르코프 체인 - 간단한 랜덤 워크
def random_walk(steps, start=0):
    positions = [start]
    for _ in range(steps):
        step = np.random.choice([-1, 1])  # -1(왼쪽) 또는 1(오른쪽)으로 이동
        positions.append(positions[-1] + step)
    return positions

# 랜덤 워크 시뮬레이션
steps = 100
rw_positions = random_walk(steps)

# 랜덤 워크 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(steps + 1), rw_positions, label="Random Walk Path")
plt.title("Example 1: Markov Chain - Random Walk")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
plt.grid()
plt.show()

# 2. 예시 2: 몬테카를로 - 원주율 추정
n_points = 100000  # 난수 점 개수

x_points = np.random.uniform(-1, 1, n_points)  # x 좌표
y_points = np.random.uniform(-1, 1, n_points)  # y 좌표

# 원 내부 점 계산
distances = x_points**2 + y_points**2
inside_circle = distances <= 1

# 원주율 계산
pi_estimate = 4 * np.sum(inside_circle) / n_points

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

# 3. 예시 3: MCMC - 가우시안 분포 샘플링
# 가우시안 파동함수 정의
def wave_function(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2))

def probability_density(x, sigma):
    psi = wave_function(x, sigma)
    return psi**2

def mcmc_sampling(n_samples, sigma, x_init, step_size):
    samples = [x_init]  # 샘플 리스트 초기화
    current_x = x_init
    
    for _ in range(n_samples):
        # 제안된 다음 샘플
        proposed_x = current_x + np.random.uniform(-step_size, step_size)
        
        # 현재와 제안된 확률밀도 계산
        current_prob = probability_density(current_x, sigma)
        proposed_prob = probability_density(proposed_x, sigma)
        
        # 채택 확률 계산
        acceptance_ratio = min(1, proposed_prob / current_prob)
        
        # 채택 여부 결정
        if np.random.uniform(0, 1) < acceptance_ratio:
            current_x = proposed_x  # 제안된 샘플을 채택
        
        samples.append(current_x)
    
    return np.array(samples)

# MCMC 샘플링 실행
n_samples = 50000
sigma = 1.0
x_init = 0.0
step_size = 0.5
mcmc_samples_example = mcmc_sampling(n_samples, sigma, x_init, step_size)

# 가우시안 이론 곡선 계산
x_vals = np.linspace(-5, 5, 1000)
y_vals = probability_density(x_vals, sigma)

# 시각화
plt.figure(figsize=(10, 6))
plt.hist(mcmc_samples_example, bins=100, density=True, alpha=0.6, label="MCMC Samples")
plt.plot(x_vals, y_vals / np.trapz(y_vals, x_vals), label="Theoretical $|\psi(x)|^2$", linewidth=2)
plt.title("Example 3: Markov Chain Monte Carlo (MCMC) - Sampling Gaussian Distribution")
plt.xlabel("$x$")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
