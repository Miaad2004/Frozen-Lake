from source import FrozenLake
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration

def get_algorithm_choice():
    while True:
        choice = input("Choose algorithm [1: Value Iteration, 2: Policy Iteration]: ")
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")

def main():
    max_iter_number = 1000000000000
       # Get algorithm choice
    choice = get_algorithm_choice()
    env = FrozenLake(render_mode="human", map_name="8x8")
    observation, info = env.reset(seed=30)
    terminal_state = env.nS - 1
    
 
    
    # Initialize chosen algorithm
    if choice == '1':
        print("\nUsing Value Iteration")
        solution = ValueIteration(env, terminal_state=terminal_state)
    else:
        print("\nUsing Policy Iteration")
        solution = PolicyIteration(env, terminal_state=terminal_state)
    
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
