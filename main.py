import ale_py
import gymnasium as gym
from tqdm import tqdm

from LearningVisualisation import LearningVisualisation
from QLearningAgent import QLearningAgent

if __name__ == '__main__':
    learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
    n_episodes = 100_000  # Number of episodes
    start_epsilon = 1.0  # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1  # Always keep some exploration

    # Create the environment
    env = gym.make("ALE/Tetris-v5", render_mode="human", obs_type="grayscale")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

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

    print(f"Using following environment settings: {env.metadata}")

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

    for episode in tqdm(range(n_episodes)):
        # Start a new game
        obs, info = env.reset()

        end_of_episode = False

        while not end_of_episode:
            # Agent chooses an action (initially random, gradually improves)
            action = agent.get_action(obs)

            # Take the action and observe the result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            end_of_episode = terminated or truncated
            obs = next_obs

        # Reduce the exploration rate (so the agent can become less random over time)
        agent.decay_epsilon()

        # Update graphs
        visualisation.update(agent)

    env.close()