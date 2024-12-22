from source import FrozenLake
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from monte_carlo import MonteCarlo
from q_learning import QLearning

def get_algorithm_choice():
    while True:
        choice = input("Choose algorithm [1: Value Iteration, 2: Policy Iteration, 3: Monte Carlo, 4: Q-Learning]: ")
        if choice in ['1', '2', '3', '4']:
            return choice
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

def main():
    max_iter_number = 1000000000000
    choice = get_algorithm_choice()
    env = FrozenLake(render_mode="human", map_name="8x8")
    observation, info = env.reset(seed=30)
    terminal_state = env.nS - 1
    
    # Initialize chosen algorithm
    if choice == '1':
        print("\nUsing Value Iteration")
        solution = ValueIteration(env, terminal_state=terminal_state)
    elif choice == '2':
        print("\nUsing Policy Iteration")
        solution = PolicyIteration(env, terminal_state=terminal_state)
    elif choice == '3':
        print("\nUsing Monte Carlo")
        solution = MonteCarlo(env, terminal_state=terminal_state)
    elif choice == '4':
        print("\nUsing Q-Learning")
        solution = QLearning(env, terminal_state=terminal_state)
    
    policy = solution.solve()
    current_state = observation

    for __ in range(max_iter_number):
        if current_state == terminal_state:
            break
        
        action = policy[current_state]
        if action == -1:
            continue
            
        next_state, reward, done, truncated, info = env.step(action)
        current_state = next_state
        won = current_state == terminal_state

        if done or won:
            observation, info = env.reset()
            current_state = observation
            
            if won:
                print("Episode won!")
                prompt = input("Continue? [Y/n]: ")
                if prompt.lower() == "n":
                    break

    env.close()

if __name__ == "__main__":
    main()
