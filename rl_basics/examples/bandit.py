import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    
    def __init__(self, K: int):
        self.K = K
        self.best_idx: int | None = None
    
    def step(self, k: int):
        raise NotImplemented

class BernoulliBandit(Bandit):
    
    def __init__(self, K: int):
        super().__init__(K)
        self.probs = np.random.uniform(size = K)
        self.best_idx = int(np.argmax(self.probs))
        self.best_prob = self.probs[self.best_idx]
        
    def step(self, k: int):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0.
        self.actions = []
        self.regrets = []
        
    def update_regret(self, k: int):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
        
    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps: int):
        
        for i in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([float(init_prob)] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
    
class DecayingEpsilonGreedy(EpsilonGreedy):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon: float = 1.0, init_prob=1.0):
        super().__init__(bandit=bandit, epsilon=epsilon, init_prob=init_prob)
        self.total_count: int = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < self.epsilon / np.sqrt(self.total_count):  
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k
    
class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k
    
def plot_results(solvers: Solver, solver_names: list[str]):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.savefig("regret.png")  


    
# ===
# test of multi-arm bandit
# ===

def test_bandit():
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print(f"There is a {K} arm bandit")
    print(f"The best arm is {bandit_10_arm.best_idx}, Probabiliry to get a reward is {bandit_10_arm.best_prob}")
    
    
def test_epsilon_greedy_solver():
    np.random.seed(1)
    bandit = BernoulliBandit(10)
    epsilon_greedy_solver = EpsilonGreedy(bandit = bandit, epsilon = 0.1, init_prob = 1)
    epsilon_greedy_solver.run(1000)
    print(f"Cumulative regret is {epsilon_greedy_solver.regret}\n")
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    print(f"The expect value of arms: {epsilon_greedy_solver.estimates}\n")
    print(f"Actions: {epsilon_greedy_solver.actions[:20]}")
    
def test_decaying_epsilon_greedy_solver():
    np.random.seed(1)
    bandit = BernoulliBandit(10)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit = bandit, epsilon = 1.0, init_prob = 1)
    decaying_epsilon_greedy_solver.run(5000)
    print(f"Cumulative regret is {decaying_epsilon_greedy_solver.regret}\n")
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
    print(f"The expect value of arms: {decaying_epsilon_greedy_solver.estimates}\n")
    print(f"Actions: {decaying_epsilon_greedy_solver.actions[:20]}")   
    
def test_ucb_solver():
    np.random.seed(1)
    bandit = BernoulliBandit(10)
    ucb_solver = UCB(bandit=bandit, coef=1.0, init_prob=1.0)
    ucb_solver.run(5000)
    print(f"Cumulative regret is {ucb_solver.regret}\n")
    plot_results([ucb_solver], ["UCB"])
    print(f"The expect value of arms: {ucb_solver.estimates}\n")
    print(f"Actions: {ucb_solver.actions[:20]}")   
    
def test_thompson_solver():
    np.random.seed(1)
    bandit = BernoulliBandit(10)
    thompson_solver = ThompsonSampling(bandit=bandit)
    thompson_solver.run(5000)
    print(f"Cumulative regret is {thompson_solver.regret}\n")
    plot_results([thompson_solver], ["ThompsonSampling"])
    print(f"Actions: {thompson_solver.actions[:20]}")   
    
    
def test_all_solver():
    bandit = BernoulliBandit(10)
    epsilon_greedy_solver = EpsilonGreedy(bandit = bandit, epsilon = 0.1, init_prob = 1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit = bandit, epsilon = 1.0, init_prob = 1)
    ucb_solver = UCB(bandit=bandit, coef=1.0, init_prob=1.0)
    thompson_solver = ThompsonSampling(bandit=bandit)
    epsilon_greedy_solver.run(5000)
    decaying_epsilon_greedy_solver.run(5000)
    ucb_solver.run(5000)
    thompson_solver.run(5000)
    plot_results([epsilon_greedy_solver, decaying_epsilon_greedy_solver, ucb_solver, thompson_solver],
                 ["EpsionGreedy", "DecayingEpsionGreedy", "UCB", "Thompson"])
    
if __name__ == "__main__":
    #test_bandit()
    #test_epsilon_greedy_solver()
    #test_decaying_epsilon_greedy_solver()
    #test_ucb_solver()
    #test_thompson_solver()
    test_all_solver()