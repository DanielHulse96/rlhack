import ale_py
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from LearningVisualisation import LearningVisualisation
from QLearningAgent import QLearningAgent

def convert_observation(observation: np.ndarray) -> tuple:
    return tuple(map(tuple, observation))

if __name__ == '__main__':

    training_mode = True # True to train, False to test
    show_gui = True

    # parameters for the agent
    learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
    n_episodes = 100_000  # Number of episodes
    start_epsilon = 1.0  # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1  # Always keep some exploration

    # Create the environment
    render_mode = "human" if show_gui else "rgb_array"
    env = gym.make("ALE/Tetris-v5", render_mode=render_mode, obs_type="grayscale")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    env.metadata["render_fps"] = 30
    print(f"Using following environment settings: {env.metadata}")

    # For visualisation:
    visualisation = LearningVisualisation(env, 500)

    # Create the agent
    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    '''
        For Tetris-v5 the action Space is Discrete(5) with the following actions
        
        * 0 -> NOOP
        * 1 -> FIRE
        * 2 -> RIGHT
        * 3 -> LEFT
        * 4 -> DOWN
    '''
    print(f"Action space: {env.action_space}")

    '''
        With obs_type set to "grayscale" the observation space is:
            Box(0, 255, (210, 160), np.uint8)
    '''
    print(f"Observation space: {env.observation_space}")

    old_epsilon = agent.epsilon
    total_rewards = []
    if training_mode:
        print("Running training mode...")
    else:
        print("Running in testing mode...")
        # If testing, temporarily disable exploration - just pure exploitation
        agent.epsilon = 0.0

    for episode in tqdm(range(n_episodes)):
        # Start a new game
        obs, info = env.reset()

        # Keep track of how many rewards, for testing
        episode_reward = 0

        end_of_episode = False
        while not end_of_episode:
            # Agent chooses an action (initially random, gradually improves)
            action = agent.get_action(convert_observation(obs))

            # Take the action and observe the result
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Learn from this experience
            agent.update(convert_observation(obs), action, reward, terminated, convert_observation(next_obs))

            # Move to next state
            end_of_episode = terminated or truncated
            obs = next_obs

        if training_mode:
            # Reduce the exploration rate (so the agent can become less random over time)
            agent.decay_epsilon()
        else:
            total_rewards.append(episode_reward)

        # Update graphs
        visualisation.update(agent)

    if training_mode:
        print(f"Finished training for {n_episodes} episodes")
    else:
        # For testing, restore old epsilon and output some metrics
        agent.epsilon = old_epsilon
        average_reward = np.mean(total_rewards)
        print(f"Test Results over {n_episodes} episodes:")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")

    env.close()