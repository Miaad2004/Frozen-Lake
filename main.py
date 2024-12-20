from source import FrozenLake
from value_iteration import ValueIteration

# Create an environment
max_iter_number = 1000000000000
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = FrozenLake(render_mode="human", map_name="8x8")
observation, info = env.reset(seed=30)

terminal_state = env.nS - 1
solution = ValueIteration(env, terminal_state=terminal_state)
policy = solution.solve()
current_state = observation

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
            print("Episode won!")
            prompt = input("Continue? [Y/n]: ")
            if prompt.lower() == "n":
                break

# Close the environment
env.close()
