import gymnasium as gym  # Import the Gymnasium library for reinforcement learning environments
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import pickle  # Import pickle for saving/loading the Q-table

def run(episodes, is_training=True, render=False):
    # Create the Taxi environment
    # 'Taxi-v3': The agent needs to pick up and drop off a passenger at specified locations
    # 'render_mode': 'human' if rendering is enabled, else None
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        # Initialize Q-table with zeros: 500 states and 6 actions
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # Load a pre-trained Q-table from file if not training
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    
    # Hyperparameters
    learning_rate_a = 0.9  # Alpha: learning rate, controls how much new info overrides old info
    discount_factor_g = 0.9  # Gamma: discount factor, measures the importance of future rewards

    epsilon = 1  # Epsilon for epsilon-greedy policy; starts with 100% exploration
    epsilon_decay_rate = 0.0001  # Rate at which epsilon decays after each episode
    rng = np.random.default_rng()  # Random number generator instance

    rewards_per_episode = np.zeros(episodes)  # Track rewards for each episode

    # Loop through each episode
    for i in range(episodes):
        # Reset the environment to get the initial state (0-499)
        state = env.reset()[0]
        terminated = False  # Flag to check if the episode has ended by reaching the destination
        truncated = False  # Flag to check if the episode ended due to a step limit

        # Continue until the agent reaches the goal or exceeds the step limit
        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                # Exploration: choose a random action if a random number < epsilon
                action = env.action_space.sample()
            else:
                # Exploitation: choose the action with the highest Q-value for the current state
                action = np.argmax(q[state, :])

            # Take the action and observe the new state and reward
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                # Update Q-value for the current state-action pair using the Q-learning formula
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
            
            # Move to the new state
            state = new_state
    
        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # When epsilon reaches 0, reduce the learning rate to avoid drastic updates
        if epsilon == 0:
            learning_rate_a = 0.001

        # Track rewards; reward is 20 if the agent successfully drops off the passenger
        if reward == 20:
            rewards_per_episode[i] = 1
    
    # Close the environment
    env.close()

    # Calculate cumulative rewards over the last 100 episodes for smoothing
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    
    # Plot the cumulative rewards and save as an image
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    # Save the Q-table to a file if training
    if is_training:
        f = open("taxi.pkl", "wb")
        pickle.dump(q, f)
        f.close()

# Entry point to run the script
if __name__ == '__main__':
    run(15000)  # Train the agent for 15,000 episodes
    run(10, is_training=False, render=True)  # Run the trained agent for 10 episodes to visualize the performance
