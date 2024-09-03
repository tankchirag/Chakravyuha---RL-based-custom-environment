"""Import required libraries for Custom environment Chakravyuh."""

import gymnasium
import numpy as np
import pygame
from pathlib import Path
import random

class ChakravyuhEnv(gymnasium.Env):
    """Custom environment representing the Chakravyuh maze."""

    def __init__(self):
        """
        Initialize the Chakravyuh environment.
        Defines grid size, agent's position, goal position, obstacles, and their powers.
        """        
        self.grid_size = (7, 7)
        self.action_space = gymnasium.spaces.Discrete(4)
        self.agent_position = np.array([0, 0])
        self.goal_position = np.array([3, 3])
        
        self.obstacle_positions = [
            {"name": "Karn", "position": np.array([2, 1])}, 
            {"name": "Balram", "position": np.array([1, 3])}, 
            {"name": "Shakuni", "position": np.array([1, 5])}, 
            {"name": "Ashwatthama", "position": np.array([3, 1])}, 
            {"name": "Drona", "position": np.array([5, 1])}, 
            {"name": "Bhishma", "position": np.array([5, 3])},
            {"name": "Kritavarma", "position": np.array([5, 5])}, 
            {"name": "Jayadratha", "position": np.array([3, 5])},
            {"name": "Kripa", "position": np.array([1, 2])},
            {"name": "Dushasan", "position": np.array([4, 4])},
            {"name": "Vikarna", "position": np.array([2, 4])}
        ]
        
        self._max_episode_steps = 100
        self._current_step = 0
        self.cumulative_reward = 0
        self.maximum_negative_reward_limit = -300
        
        self.obstacles = {
            "Ashwatthama": {"penalty": 20, "power": "penalty"},
            "Kritavarma": {"penalty": 10, "power": "penalty"},
            "Shakuni": {"teleport_random": True, "power": "teleport_random"},
            "Bhishma": {"effect": 4, "power": "freeze"},
            "Drona": {"teleport_position": np.array([0, 0]), "power": "teleport"},
            "Karn": {"health_reduction": 1, "power": "health_reduction"},
            "Balram": {"score_multiplier": 4, "power": "score_multiplier"},
            "Jayadratha": {"restrict_left": (np.array([3, 5]), np.array([3, 3])), "power": "restrict_left"},
            "Kripa": {"effect": 2, "power": "freeze"},
            "Dushasan": {"teleport_position": np.array([6, 6]), "power": "teleport"},
            "Vikarna": {"power": "game_over"}
        }

        self.health = 3
        self.score_multiplier = 1.0

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            np.array: The initial position of the agent.
        """        
        self.agent_position = np.array([0, 0])
        self._current_step = 0
        self.cumulative_reward = 0
        self.freeze_steps = 0
        self.health = 3
        self.score_multiplier = 1.0
        self.restricted_left = False
        return self.agent_position

    def step(self, action):
        """
        Execute a step in the environment based on the given action.

        Args:
            action (int): The action to be taken by the agent.

        Returns:
            np.array: New position of the agent.
            float: Cumulative reward.
            bool: Whether the episode is done.
            dict: Additional information.
        """
        initial_position = self.agent_position.copy()
        first_time = self._current_step == 0

        # Handle freeze steps
        if self.freeze_steps > 0:
            print(f"Agent is frozen for {self.freeze_steps} more steps.")
            self.freeze_steps -= 1
            self._current_step += 1
            return self.agent_position, self.cumulative_reward, False, {}

        # Handle restricted left movement by Jayadratha
        if self.restricted_left and action == 2 and (self.agent_position[0] == 3 and 3 <= self.agent_position[1] <= 5):
            print("Left movement restricted by Jayadratha.")
            self._current_step += 1
            return self.agent_position, self.cumulative_reward, False, {}

        # Update the agent's position based on the action
        action_mapping = {0: np.array([-1, 0]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([0, 1])}
        new_position = self.agent_position + action_mapping[action]

        if not self._is_within_bounds(new_position):
            new_position = self.agent_position  # Agent stays in the same position if out of bounds
        else:
            obstacle = next((obs for obs in self.obstacle_positions if (new_position == obs["position"]).all()), None)
            if obstacle:
                obstacle_name = obstacle["name"]
                obstacle_power = self.obstacles[obstacle_name]["power"]
                result = self._handle_obstacle(obstacle_name, obstacle_power, new_position)
                if result is not None:
                    return result

        self.agent_position = new_position
        reward = self._calculate_reward()
        self.cumulative_reward += reward
        done = (self.agent_position == self.goal_position).all()
        self._current_step += 1

        # Check for end conditions
        if self.cumulative_reward <= self.maximum_negative_reward_limit:
            print(f"Cumulative reward reached maximum limit {self.maximum_negative_reward_limit}.")
            done = True
        elif self._current_step >= self._max_episode_steps:
            done = True
            if not (self.agent_position == self.goal_position).all():
                print("Abhimanyu is trapped in the Chakravyuh.")
        elif done:
            print("Abhimanyu has reached Duryodhan.")
            print(f"New State: {self.agent_position}, Cumulative Reward: {self.cumulative_reward}, Distance to Goal: [0 0], Required no. of actions: 0")

        required_action = np.sum(np.abs(self.agent_position - self.goal_position))
        distance_to_goal = np.array(self.agent_position - self.goal_position)

        if first_time:
            print(f"Initial State: {initial_position}")
            print(f"New State: {self.agent_position}, Cumulative Reward: {self.cumulative_reward}, Distance to Goal: {distance_to_goal}, Required no. of actions: {required_action}")
        elif not done:  # Only print the new state if the game is not done
            print(f"New State: {self.agent_position}, Cumulative Reward: {self.cumulative_reward}, Distance to Goal: {distance_to_goal}, Required no. of actions: {required_action}")

        return self.agent_position, self.cumulative_reward, done, {}


    def render(self, screen):
        """
        Render the environment using images of the agent, goal, and obstacles.

        Args:
            screen (pygame.Surface): The Pygame surface to draw the environment on.
        """
        background = pygame.image.load(str(Path("images") / "chkravyuha.jpeg"))
        background = pygame.transform.scale(background, (self.grid_size[1] * 100, self.grid_size[0] * 100))
        screen.blit(background, (0, 0))

        agent_img = pygame.image.load(str(Path("images") / "abhimanyu.jpeg"))
        agent_img = pygame.transform.scale(agent_img, (100, 100))
        goal_img = pygame.image.load(str(Path("images") / "duryodhan.jpeg"))
        goal_img = pygame.transform.scale(goal_img, (100, 100))

        obstacle_imgs = {}
        for obstacle in self.obstacle_positions:
            obstacle_imgs[obstacle["name"]] = pygame.image.load(str(Path("images") / f'{obstacle["name"].lower()}.jpeg'))
            obstacle_imgs[obstacle["name"]] = pygame.transform.scale(obstacle_imgs[obstacle["name"]], (100, 100))

        screen.blit(agent_img, (self.agent_position[1] * 100, self.agent_position[0] * 100))
        screen.blit(goal_img, (self.goal_position[1] * 100, self.goal_position[0] * 100))

        for obstacle in self.obstacle_positions:
            screen.blit(obstacle_imgs[obstacle["name"]], (obstacle["position"][1] * 100, obstacle["position"][0] * 100))

    def _is_within_bounds(self, position):
        """
        Check if a position is within the grid bounds.

        Args:
            position (np.array): The position to check.

        Returns:
            bool: True if within bounds, False otherwise.
        """
        return (0 <= position[0] < self.grid_size[0]) and (0 <= position[1] < self.grid_size[1])
    
    def _calculate_reward(self):
        """
        Calculate reward based on the agent's current position.

        Returns:
            float: The calculated reward.
        """
        if (self.agent_position == self.goal_position).all():
            return 400
        elif any((self.agent_position == obs["position"]).all() for obs in self.obstacle_positions):
            return -5 * self.score_multiplier
        else:
            return 0

    def _handle_obstacle(self, obstacle_name, obstacle_power, new_position):
        """Handle interactions with obstacles.
        
        Args:
            obstacle_name (str): Name of the obstacle.
            obstacle_power (str): Power of the obstacle.
            new_position (np.array): The new position of the agent.
        
        Returns:
            Tuple: Updated agent position, cumulative reward, done status, and info dictionary.
        """        
        if obstacle_power == "penalty":
            penalty = self.obstacles[obstacle_name]["penalty"]
            print(f"Abhimanyu has been additionally penalized by {obstacle_name} by {penalty} points.")
            self.cumulative_reward -= penalty * self.score_multiplier
        elif obstacle_power == "freeze":
            self.freeze_steps = self.obstacles[obstacle_name]["effect"]
            print(f"Abhimanyu has been frozen by {obstacle_name} for {self.freeze_steps} steps.")
        elif obstacle_power == "health_reduction":
            penalty = (self.maximum_negative_reward_limit-self.cumulative_reward) / self.health if self.health > 0 else self.maximum_negative_reward_limit
            self.health -= self.obstacles[obstacle_name]["health_reduction"]
            print(f"Abhimanyu's health power reduced by {obstacle_name}. Current health: {self.health}")            
            self.cumulative_reward += penalty
            print(f"Karn's additional reward penalty applied: {penalty}.")
        elif obstacle_power == "teleport":
            new_position[:] = self.obstacles[obstacle_name]["teleport_position"]
            print(f"Abhimanyu has been teleported by {obstacle_name} to position {new_position}.")
            self.cumulative_reward -= 5 * self.score_multiplier  # Apply multiplier for teleport obstacles
        elif obstacle_power == "score_multiplier":
            self.score_multiplier = self.obstacles[obstacle_name]["score_multiplier"]
            print(f"Abhimanyu's score multiplier changed by {obstacle_name} to {self.score_multiplier}.")
        elif obstacle_power == "restrict_left":
            start, end = self.obstacles[obstacle_name]["restrict_left"]
            if start[0] == end[0] and start[0] == self.agent_position[0]:
                self.restricted_left = True
                print(f"Abhimanyu's left movement restricted by {obstacle_name} from position {start} to {end}.")
        elif obstacle_power == "teleport_random":
            new_position[:] = np.array([random.randint(0, 6), random.randint(0, 6)])
            print(f"Abhimanyu has been randomly teleported by {obstacle_name} to position {new_position}.")
            self.cumulative_reward -= 5 * self.score_multiplier  # Apply multiplier for teleport obstacles
        elif obstacle_power == "game_over":
            vikarna_penalty = (self.maximum_negative_reward_limit-self.cumulative_reward)
            self.cumulative_reward += vikarna_penalty
            print(f"Abhimanyu has encountered by {obstacle_name}'s additional penalty {vikarna_penalty}, Cumulative Reward: {self.cumulative_reward}. Game over.")
            return self.agent_position, self.cumulative_reward, True, {}

# Pygame for rendering
pygame.init()

# Set up Pygame display
WINDOW_SIZE = (700, 700)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Chakravyuh Visualization")
clock = pygame.time.Clock()

# Load and play the background sound
pygame.mixer.init()
pygame.mixer.music.load('sound.mp3')  
pygame.mixer.music.play(-1)  # Loop the sound indefinitely

# Create Chakravyuh environment instance
env = ChakravyuhEnv()

# Reset the environment to the initial state
env.reset()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 0  # Up
            elif event.key == pygame.K_DOWN:
                action = 1  # Down
            elif event.key == pygame.K_LEFT:
                action = 2  # Left
            elif event.key == pygame.K_RIGHT:
                action = 3  # Right
            else:
                continue  # Ignore other keys

            # Perform the action in the environment
            observation, cumulative_reward, done, _ = env.step(action)

            # Check if episode is done
            if done:
                running = False

    # Render the environment
    env.render(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
pygame.mixer.quit()  # Properly quit the mixer
