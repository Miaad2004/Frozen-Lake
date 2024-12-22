import numpy as np
from solution import Solution
from tqdm import tqdm
from collections import deque
import random

class QLearning(Solution):
    def __init__(self, env, gamma=0.95, lr=0.1, epsilon=0.9, epsilon_decay=0.8,
                 min_epsilon=0.05, n_episodes=50000, max_episode_steps=500,
                 terminal_state=None):
        self.env = env
        self.terminal_state = terminal_state if terminal_state else env.nS - 1
        
        self.gamma = gamma
        self.lr = lr
        self.training = True
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.episodes = n_episodes
        self.max_episode_steps = max_episode_steps
        
        # Double Q-learning tables
        self.Q1 = np.zeros((env.nS, env.nA))
        self.Q2 = np.zeros((env.nS, env.nA))
        self.pi = np.zeros(env.nS)

        # visit counter
        self.visit_counter = {}
        self.revisit_penalty = -1

    def act(self, current_state):
        if self.training and np.random.random() < self.epsilon:
            return np.random.randint(self.env.nA)
        
        return np.argmax((self.Q1[current_state] + self.Q2[current_state]) / 2)

    def reward(self, state, action, next_state):
        base_reward = -0.1
        if next_state == self.terminal_state:
            return 1000.0
        
        transmission = self.env.P[state][action][0]
        is_hole = transmission[3]
        if is_hole:
            return -30.0
        
        # Add revisit penalty
        revisit_penalty = self.revisit_penalty * self.visit_counter.get(next_state, 0)
        
        # Simplified hole avoidance - check nearby states
        hole_penalty = 0
        for a in range(self.env.nA):
            for prob, next_s, _, is_hole in self.env.P[state][a]:
                if is_hole:
                    hole_penalty -= 1.0
                    break
        
        # Distance-based reward
        grid_size = int(np.sqrt(self.env.nS))
        s_row, s_col = divmod(state, grid_size)
        t_row, t_col = divmod(self.terminal_state, grid_size)
        distance = abs(s_row - t_row) + abs(s_col - t_col)
        distance_reward = -10 * distance
        
        return base_reward + hole_penalty + revisit_penalty + distance_reward

    def update_q_value(self, state, action, reward, next_state):
        if np.random.random() < 0.5:
            old_q_value = self.Q1[state][action]
            best_next_action = np.argmax(self.Q1[next_state])
            next_q_value = self.Q2[next_state][best_next_action]
            
            td_target = reward + self.gamma * next_q_value
            td_error = td_target - old_q_value
            
            self.Q1[state][action] += self.lr * td_error
            
        else:
            old_q_value = self.Q2[state][action]
            best_next_action = np.argmax(self.Q2[next_state])
            next_q_value = self.Q1[next_state][best_next_action]
            
            td_target = reward + self.gamma * next_q_value
            td_error = td_target - old_q_value

            self.Q2[state][action] += self.lr * td_error

    def on_episode_end(self, episode):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax((self.Q1 + self.Q2) / 2, axis=1)

    def solve(self):
        print("Training Q-Learning agent...")
        step_counter = 0
        
        for episode in tqdm(range(self.episodes)):
            state, _ = self.env.reset()
            
            # Reset visit counter for new episode
            self.visit_counter = {}
            
            for step in range(self.max_episode_steps):
                action = self.act(state)
                next_state, _, done, truncated, _ = self.env.step(action)
                
                # Update visit counter
                self.visit_counter[next_state] = self.visit_counter.get(next_state, 0) + 1
                
                reward = self.reward(state, action, next_state)
                self.update_q_value(state, action, reward, next_state)
                
                state = next_state
                step_counter += 1
                
                if state == self.terminal_state:
                    break
                    
                if done or truncated:
                    break
                
            self.on_episode_end(episode)
        
        self.training = False
        print("Training complete.")
        return self.get_policy()