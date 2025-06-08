import gymnasium as gym
import numpy as np
import time

# --- 1. 初始化环境 ---
# is_slippery=True 表示环境是随机的（在冰上走可能会滑到其他格子）
# 这是 FrozenLake 的标准模式，更能体现策略迭代的价值。
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)

# 获取状态和动作的数量
num_states = env.observation_space.n
num_actions = env.action_space.n

print(f"状态空间数量 (Number of states): {num_states}")
print(f"动作空间数量 (Number of actions): {num_actions}")

def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    策略迭代算法实现

    Args:
        env: Gymnasium 环境.
        gamma (float): 折扣因子.
        theta (float): 策略评估的收敛阈值.

    Returns:
        tuple: (最优策略, 最优价值函数).
    """
    # --- 算法初始化 ---
    policy = np.random.randint(num_actions, size=num_states)
    V = np.zeros(num_states)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- 迭代轮次: {iteration} ---")
        
        # --- 2. 策略评估 (Policy Evaluation) ---
        print("1. 正在进行策略评估...")
        while True:
            delta = 0
            for s in range(num_states):
                v_old = V[s]
                action = policy[s]
                # 【修正】使用 env.unwrapped.P 访问转移模型
                V[s] = sum([prob * (reward + gamma * V[next_state])
                            for prob, next_state, reward, _ in env.unwrapped.P[s][action]])
                delta = max(delta, abs(v_old - V[s]))
            
            if delta < theta:
                print("   策略评估完成，价值函数已收敛。")
                break
        
        # --- 3. 策略改进 (Policy Improvement) ---
        print("2. 正在进行策略改进...")
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            
            action_values = np.zeros(num_actions)
            for a in range(num_actions):
                # 【修正】使用 env.unwrapped.P 访问转移模型
                action_values[a] = sum([prob * (reward + gamma * V[next_state])
                                        for prob, next_state, reward, _ in env.unwrapped.P[s][a]])
            
            best_action = np.argmax(action_values)
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            print("   策略改进完成，策略已稳定。找到最优策略！")
            break
            
    return policy, V

# --- 训练 Agent ---
print("开始使用策略迭代算法进行训练...")
optimal_policy, optimal_value_function = policy_iteration(env)

print("\n--- 训练完成 ---")
print("最优策略 (0:左, 1:下, 2:右, 3:上):")
print(optimal_policy.reshape(4, 4))
print("\n最优价值函数:")
print(optimal_value_function.reshape(4, 4))


# --- 4. 使用训练好的策略与环境交互并渲染 ---
print("\n--- 开始使用最优策略进行演示 (共 3 个回合) ---")
for episode in range(3):
    state, info = env.reset()
    terminated = False
    truncated = False
    print(f"\n--- 第 {episode + 1} 回合 ---")
    time.sleep(1) # 暂停1秒，方便观察
    
    while not terminated and not truncated:
        # 不需要再探索，直接使用最优策略
        action = optimal_policy[state]
        
        # 与环境交互
        state, reward, terminated, truncated, info = env.step(action)
        
        # env.render() 会在 step 后自动调用 (因为 render_mode="human")
        time.sleep(0.3) # 每一步稍作暂停，方便观察
        
    if terminated and reward == 1.0:
        print("成功到达终点！🎉")
    else:
        print("掉进冰窟或超时。😭")

# --- 5. 关闭环境 ---
env.close()