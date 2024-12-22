import numpy as np
from solution import Solution

class MonteCarlo(Solution):
    def __init__(self, env, gamma=0.9, num_episodes=1000, terminal_state=None):
        self.env = env
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.terminal_state = terminal_state if terminal_state else env.nS - 1
        
        self.V = np.zeros(env.nS)
        self.Q = np.zeros((env.nS, env.nA))
        self.returns = {(s, a): [] for s in range(env.nS) for a in range(env.nA)}
        self.pi = np.random.randint(0, env.nA, size=env.nS)
        self.pi[self.terminal_state] = -1

    def generate_episode(self):
        episode = []
        state = self.env.reset()[0]
        
        while True:
            action = self.pi[state]
            if action == -1:
                break
                
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            
            if done or state == self.terminal_state:
                break
                
            state = next_state
            
        return episode

    def monte_carlo_evaluation(self):
        for _ in range(self.num_episodes):
            episode = self.generate_episode()
            G = 0
            visited_pairs = set()
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                if (state, action) not in visited_pairs:
                    visited_pairs.add((state, action))
                    self.returns[(state, action)].append(G)
                    self.Q[state, action] = np.mean(self.returns[(state, action)])

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

    def print_policy(self):
        action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT', -1: 'GOAL'}
        policy_grid = np.array(self.pi).reshape(self.env.shape)
        policy_grid = np.vectorize(lambda x: action_names.get(x, 'NONE'))(policy_grid)
        
        print("\nOptimal Policy:")
        for row in policy_grid:
            print(" ".join(f"{action:5}" for action in row))

    def solve(self):
        iteration = 0
        while True:
            iteration += 1
            print(f"\nIteration {iteration}")
            self.monte_carlo_evaluation()
            if self.policy_improvement():
                break
        
        self.print_policy()
        return self.pi