from LearningVisualisation import LearningVisualisation
import gymnasium as gym
import os
import dill as pickle

class SnapshotManager:
    def __init__(self,
                 env : gym.Env):
        self.env = env
        self.game_name = env.spec.id.split("/")[-1].split("-")[0].lower()

    def save_agent(self, agent, episode):
        folder_path = os.path.join("trained_agents", self.game_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"trained_agent_{episode}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump({
                "q_values": agent.q_values,
                "epsilon": agent.epsilon,
                "training_error": agent.training_error
            }, f)
        print(f"Agent saved to {file_path}")

    def restore_agent_state(self, agent, episode):
        file_path = os.path.join("trained_agents", self.game_name, f"trained_agent_{episode}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                agent.q_values = data["q_values"]
                agent.epsilon = data["epsilon"]
                agent.training_error = data["training_error"]  # Load the entire agent object
            print(f"Loaded trained agent from episode {episode}")
            return agent
        else:
            print(f"No trained agent found for episode {episode}")
            return None