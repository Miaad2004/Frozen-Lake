# q_learning.py
import numpy as np
from solution import Solution

class QLearning(Solution):
    def __init__(self, env, gamma=0.9, alpha=0.5, epsilon=0.5, episodes=100, terminal_state=None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.episodes = episodes
        self.terminal_state = terminal_state if terminal_state else env.nS - 1
        
        self.Q = np.zeros((env.nS, env.nA))
        self.pi = np.zeros(env.nS)

    def epsilon_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.nA)
        return np.argmax(self.Q[state])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def print_policy(self):
        action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT', -1: 'GOAL'}
        for state in range(self.env.nS):
            self.pi[state] = np.argmax(self.Q[state]) if state != self.terminal_state else -1
        
        policy_grid = np.array(self.pi).reshape(self.env.shape)
        policy_grid = np.vectorize(lambda x: action_names.get(x, 'NONE'))(policy_grid)
        
        print("\nOptimal Policy:")
        for row in policy_grid:
            print(" ".join(f"{action:5}" for action in row))

    def solve(self):
        for episode in range(self.episodes):
            state = self.env.reset()[0]
            done = False
            
            while not done:
                action = self.epsilon_greedy_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Modify reward structure
                if next_state == self.terminal_state:
                    reward = 10
                elif done:  # Fell in hole
                    reward = -1
                else:
                    reward = -0.1  # Step cost
                
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                
                if state == self.terminal_state:
                    break
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            if episode % 1000 == 0:
                print(f"Episode {episode}/{self.episodes}")
        
        self.print_policy()
        return self.pi