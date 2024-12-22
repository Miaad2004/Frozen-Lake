from colorama import init, Fore, Back, Style
from source import FrozenLake
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from monte_carlo import MonteCarlo
from q_learning import QLearning
import numpy as np

init() 

def get_algorithm_choice():
    while True:
        print(f"\n{Fore.CYAN}Available Algorithms:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1: Value Iteration")
        print("2: Policy Iteration")
        print("3: Monte Carlo")
        print(f"4: Q-Learning{Style.RESET_ALL}")
        choice = input(f"\n{Fore.GREEN}Choose algorithm [1-4]: {Style.RESET_ALL}")
        if choice in ['1', '2', '3', '4']:
            return choice
        print(f"{Fore.RED}Invalid choice. Please enter 1, 2, 3, or 4.{Style.RESET_ALL}")

def print_policy_grid(policy_grid):
    print(f"\n{Fore.YELLOW}Optimal Policy:{Style.RESET_ALL}")
    for row in policy_grid:
        print(" ".join(f"{Back.BLUE}{Fore.WHITE}{action:5}{Style.RESET_ALL}" for action in row))

def main():
    max_iter_number = 10000
    choice = get_algorithm_choice()
    env = FrozenLake(render_mode="human", map_name="8x8", speed_multiplier=0.4)
    observation, info = env.reset(seed=30)
    terminal_state = env.nS - 1
    action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
    
    algorithm_names = {
        '1': 'Value Iteration',
        '2': 'Policy Iteration', 
        '3': 'Monte Carlo',
        '4': 'Q-Learning'
    }
    print(f"\n{Fore.CYAN}Using {algorithm_names[choice]}{Style.RESET_ALL}")
    
    if choice == '1':
        env.update_algorithm_name("Value Iteration")
        solution = ValueIteration(env, terminal_state=terminal_state)
        
    elif choice == '2':
        env.update_algorithm_name("Policy Iteration")
        solution = PolicyIteration(env, terminal_state=terminal_state)
        
    elif choice == '3':
        env.update_algorithm_name("Monte Carlo")
        env.render_mode = "ansi"
        solution = MonteCarlo(env, terminal_state=terminal_state)
        
    elif choice == '4':
        env.update_algorithm_name("Q-Learning")
        env.render_mode = "ansi"
        solution = QLearning(env, terminal_state=terminal_state)

    
    policy = solution.solve()
    
    # Print the optimal policy
    policy_grid = np.array(policy).reshape(8, 8)
    policy_grid = np.vectorize(action_names.get)(policy_grid)
    print_policy_grid(policy_grid)
        
    current_state = observation
    env.render_mode = "human"
    
    for __ in range(max_iter_number):
        if current_state == terminal_state:
            break
        
        action = policy[current_state]
            
        next_state, reward, done, truncated, info = env.step(action)
        current_state = next_state
        won = current_state == terminal_state

        if done or won:
            observation, info = env.reset()
            current_state = observation
            
            if won:
                print(f"\n{Fore.GREEN}Episode won!{Style.RESET_ALL}")
                prompt = input(f"{Fore.YELLOW}Continue? [Y/n]: {Style.RESET_ALL}")
                if prompt.lower() == "n":
                    break

    env.close()

if __name__ == "__main__":
    main()
