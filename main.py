from source import FrozenLake
# Create an environment
max_iter_number = 1000
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = FrozenLake(render_mode="human", map_name="8x8")
observation, info = env.reset(seed=30)

for __ in range(max_iter_number):
    # Note: .sample() is used to sample random action from the environment's action space

    # Choose an action (Replace this random action with your agent's policy)
    action = env.action_space.sample()

    # Perform the action and receive feedback from the environment
    next_state, reward, done, truncated, info = env.step(action)

    if done or truncated:
        observation, info = env.reset()

# Close the environment
env.close()
