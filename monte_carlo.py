import numpy as np
from solution import Solution

class MonteCarlo(Solution):
    def __init__(self, env, gamma=0.9, num_episodes=1000, epsilon=0.1, terminal_state=None):
        self.env = env
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.terminal_state = terminal_state if terminal_state else env.nS - 1
        
        self.V = np.zeros(env.nS)
        self.Q = np.zeros((env.nS, env.nA))
        self.returns = {(s, a): [] for s in range(env.nS) for a in range(env.nA)}
        self.pi = np.random.randint(0, env.nA, size=env.nS)
        self.pi[self.terminal_state] = -1
        
        # New additions for optimization
        self.min_episodes = 100
        self.convergence_threshold = 1e-9
        self.batch_size = 20

    def get_action(self, state):
        if state == self.terminal_state:
            return -1
        return np.random.randint(0, self.env.nA) if np.random.random() < self.epsilon else self.pi[state]

    def generate_episode(self):
        episode = []
        state = self.env.reset()[0]
        max_steps = 500
        
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
        step_cost = -0.2
        
        if next_state == self.terminal_state:
            return 20
        
        transmission = self.env.P[state][action][0]
        is_hole = transmission[3]
        
        if not is_hole:
             return step_cost
        
        return -5
    
    def monte_carlo_evaluation(self):
        old_Q = self.Q.copy()
        episodes = [self.generate_episode() for _ in range(self.batch_size)]
        
        for episode in episodes:
            G = 0
            visited_pairs = set()
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                if (state, action) not in visited_pairs:
                    visited_pairs.add((state, action))
                    self.returns[(state, action)].append(G)
                    # Keep only recent returns to save memory
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
                print(f"Iteration {iteration}, Episodes: {episodes_done}, Delta: {delta:.4f}")
            
            if self.policy_improvement() and episodes_done > self.min_episodes and delta < self.convergence_threshold:
                break
        
        return self.pi