import gymnasium as gym
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('FrozenLake-v1', render_mode='rgb_array') # 确保设置了 render_mode

# 重置环境以获取初始状态和信息
# observation 是代理的初始状态，info 包含额外信息
observation, info = env.reset()
rgb_array = env.render()

# 现在可以渲染环境了
plt.imshow(rgb_array)
plt.title("FrozenLake Initial State")
plt.axis("off")
plt.show()

# 如果你后续想执行动作并观察结果，可以这样做：
# action = env.action_space.sample() # 随机选择一个动作
# next_observation, reward, terminated, truncated, info = env.step(action)
# env.render()

# 关闭环境
env.close()