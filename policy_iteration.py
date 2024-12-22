import numpy as np
from solution import Solution


class PolicyIteration(Solution):
    def __init__(self, env, gamma=0.9, theta=1e-10, terminal_state=None):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.terminal_state = terminal_state

        if not terminal_state:
            self.terminal_state = env.nS - 1

        self.V = np.zeros(env.nS)
        self.pi = np.zeros(env.nS)

    def reward(self, state, action, next_state):
        if next_state == self.terminal_state:
            return 2.0  
            
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

    def probability(self, state, action, next_state):
        if self.env.is_hardmode:
            probs = {
                0: {0: 0.5, 1: 0.25, 3: 0.25},  # UP
                1: {0: 0.25, 1: 0.5, 2: 0.25},  # RIGHT
                2: {1: 0.25, 2: 0.5, 3: 0.25},  # DOWN
                3: {0: 0.25, 2: 0.25, 3: 0.5},  # LEFT
            }

            total_prob = 0
            for possible_action, prob in probs[action].items():
                transition = self.env.P[state][possible_action][0]
                possible_next_state = transition[1]

                if possible_next_state == next_state:
                    total_prob += prob

            return total_prob
        else:
            transition = self.env.P[state][action][0]
            return 1.0 if transition[1] == next_state else 0.0

    def Q(self, state, action):
        expected_reward = 0
        for next_state in range(self.env.nS):
            p = self.probability(state, action, next_state)
            r = self.reward(state, action, next_state)
            expected_reward += p * (r + self.gamma * self.V[next_state])

        return expected_reward

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.env.nS):
                if state == self.terminal_state:
                    self.V[state] = 0
                    continue

                v = self.V[state]
                self.V[state] = self.Q(state, self.pi[state])
                delta = max(delta, abs(v - self.V[state]))

            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.env.nS):
            if state == self.terminal_state:
                self.pi[state] = None
                continue

            old_action = self.pi[state]
            self.pi[state] = np.argmax(
                [self.Q(state, action) for action in range(self.env.nA)]
            )

            if old_action != self.pi[state]:
                policy_stable = False

        return policy_stable

    def solve(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

        return self.pi
