import sys

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

from LearningVisualisation import LearningVisualisation
from QLearningAgent import QLearningAgent
from SnapshotManager import SnapshotManager

resume_from_save = False
episode_to_load = 500 # Only used if resume_from_save is True
snapshot_frequency = 10 # How often to take a snapshot of the agent

training_mode = True  # True to train, False to test
show_gui = False  # True to see the live updates, false to not - faster training if False
record_video = True
record_video_frequency = 500  # How many episodes to wait before recording a video
visualisation_frequency = 500 # How often to show the graphs

# parameters for the agent
learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
n_episodes = 100_000  # Number of episodes
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1  # Always keep some exploration

def convert_observation(observation: np.ndarray) -> tuple:
    return tuple(map(tuple, observation))

if __name__ == '__main__':
    # Sanity checks
    if show_gui and record_video:
        print("show_gui must be False if record_video is True" 
              "(render_mode must be rgb_array for recording)")
        sys.exit()

    # Create the environment
    render_mode = "human" if show_gui else "rgb_array"
    env = gym.make("ALE/Tetris-v5", render_mode=render_mode, obs_type="grayscale")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    if record_video:
        env = RecordVideo(env,
                          video_folder="tetris-agent-videos",
                          name_prefix="tetris-agent",
                          episode_trigger=lambda x: x % record_video_frequency == 0)

    env.metadata["render_fps"] = 30
    print(f"Using following environment settings: {env.metadata}")

    # For visualisation:
    visualisation = LearningVisualisation(env.env, 500) if record_video else (
        LearningVisualisation(env, 500))

    # Create agent and load the previous state if boolean is set
    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon)

    snapshot_manager = SnapshotManager(env)
    if resume_from_save:
        agent = snapshot_manager.restore_agent_state(agent, episode_to_load)

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
        if episode % visualisation_frequency == 0:
            visualisation.update(agent, episode)

        #if episode % snapshot_frequency == 0:
            #snapshot_manager.save_agent(agent, episode)

    if training_mode:
        print(f"Finished training for {n_episodes} episodes")
        print(f"Final epsilon: {agent.epsilon}")
    else:
        # For testing, restore old epsilon and output some metrics
        agent.epsilon = old_epsilon
        average_reward = np.mean(total_rewards)
        print(f"Test Results over {n_episodes} episodes:")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")

    env.close()