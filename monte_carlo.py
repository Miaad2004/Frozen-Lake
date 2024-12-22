import numpy as np
from solution import Solution


class MonteCarlo(Solution):
    def __init__(self,
                 env,
                 gamma=0.9,
                 num_episodes=10000,
                 epsilon=0.35,
                 terminal_state=None):
        # Environment params
        self.env = env
        self.terminal_state = terminal_state if terminal_state else env.nS - 1
        
        # learning params
        self.gamma = gamma  # Discount factor
        self.num_episodes = num_episodes  # Number of episodes to run
        self.epsilon = epsilon  # Exploration rate
        
        # State-Action value init
        self.V = np.zeros(env.nS)  # State values
        self.Q = np.zeros((env.nS, env.nA))  # State-action values
        self.pi = np.random.randint(0, env.nA, size=env.nS)  # Random initial policy
        self.pi[self.terminal_state] = -1  
        
        # Learning tracking params
        self.returns = {(s, a): [] for s in range(env.nS) for a in range(env.nA)}
        
        # opt params
        self.min_episodes = 100  # min episodes before convergence check
        self.convergence_threshold = 1e-6
        self.batch_size = 20  

    def get_action(self, state):
        if state == self.terminal_state:
            return -1
        return (
            np.random.randint(0, self.env.nA)
            if np.random.random() < self.epsilon
            else self.pi[state]
        )

    def generate_episode(self):
        episode = []
        state = self.env.reset()[0]
        max_steps = 1000

        for _ in range(max_steps):
            action = self.get_action(state)
            if action == -1:
                break

            next_state, _, done, _, _ = self.env.step(action)
            reward = self.reward(state, action, next_state)
            episode.append((state, action, reward))

            if done:
                break

            state = next_state

        return episode

    def reward(self, state, action, next_state):
        if next_state == self.terminal_state:
            return 1.0  
            
        transmission = self.env.P[state][action][0]
        is_hole = transmission[3]
        
        if is_hole:
            return -1.0  
        
        # Distance-based reward
        grid_size = int(np.sqrt(self.env.nS))
        s_row, s_col = divmod(state, grid_size)
        t_row, t_col = divmod(self.terminal_state, grid_size)
        manhattan_dist = abs(s_row - t_row) + abs(s_col - t_col)
        
        # Reward for moving closer to goal
        next_row, next_col = divmod(next_state, grid_size)
        next_dist = abs(next_row - t_row) + abs(next_col - t_col)
        if next_dist < manhattan_dist:
            return 0.1 
            
        return -0.01  # negative reward for other moves

    def monte_carlo_evaluation(self):
        old_Q = self.Q.copy()
        episodes = [self.generate_episode() for _ in range(self.batch_size)]

        for episode in episodes:
            G = 0
            visited_pairs = set()

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward

                if (state, action) not in visited_pairs:
                    visited_pairs.add((state, action))
                    self.returns[(state, action)].append(G)
                    self.returns[(state, action)] = self.returns[(state, action)][-1000:]
                    self.Q[state, action] = np.mean(self.returns[(state, action)])

        return np.max(np.abs(old_Q - self.Q))

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.env.nS):
            if state == self.terminal_state:
                continue

            old_action = self.pi[state]
            self.pi[state] = np.argmax(self.Q[state])

            if old_action != self.pi[state]:
                policy_stable = False
        return policy_stable

    def solve(self):
        iteration = 0
        episodes_done = 0

        while episodes_done < self.num_episodes:
            iteration += 1
            delta = self.monte_carlo_evaluation()
            episodes_done += self.batch_size

            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}, Episodes: {episodes_done}, Delta: {delta:.4f}"
                )

            if (
                self.policy_improvement()
                and episodes_done > self.min_episodes
                and delta < self.convergence_threshold
            ):
                break

        return self.pi
