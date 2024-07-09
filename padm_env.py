import gymnasium as gym
import numpy as np
import pygame as py
import sys
import random

class Akp_Env(gym.Env):
    def __init__(self, goal_state, hell_state, grid_size=11):
        super(Akp_Env, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 70
        self.state = None
        self.reward = 0
        self.info = {}
        self.hell_states = hell_state
        
        self.agent_state_number = [(10, 0), (0, 10), (0, 0), (10, 10)]
        self.agent_state = random.choice(self.agent_state_number)
        
        self.goal_state = goal_state
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
        self.in_portal = (6, 5)
        self.out_portal = (2, 5)
        
        self.reversal_portal_1 = (5, 3)
        self.reversal_portal_2 = (6, 10)
        
        self.wall_states = [(1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1),
                            (4, 2), (4, 3), (4, 4), (6, 2), (6, 3), (6, 4), (4, 6), (4, 7), (4, 8), (4, 9),
                            (6, 6), (6, 7), (6, 8), (6, 9), (1, 9), (2, 9), (3, 9), (7, 9), (8, 9), (9, 9),
                            (1, 3), (1, 4), (1, 6), (1, 7), (2, 3), (2, 4), (2, 6), (2, 7),
                            (8, 3), (8, 4), (8, 6), (8, 7), (9, 3), (9, 4), (9, 6), (9, 7)]
        
        py.init()
        self.screen = py.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        py.display.set_caption("Find the Way")

        try:
            self.agent_image = py.image.load("background/agent.png").convert_alpha()
        except FileNotFoundError:
            print("Agent image not found.")
            self.agent_image = py.Surface((self.cell_size, self.cell_size))
            self.agent_image.fill((255, 255, 255))
        
        
        self.agent_image = py.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        self.agent_direction = 0

    def reset(self):
        self.agent_state = random.choice(self.agent_state_number)
        self.done = False
        self.reward = 0

        self.info["Distance to Goal"] = np.sqrt((self.agent_state[0] - self.goal_state[0]) ** 2 +
                                                (self.agent_state[1] - self.goal_state[1]) ** 2)
        
        if self.agent_state == (0, 0) or self.agent_state == (10, 10):
            self.agent_direction += 180
        else:
            self.agent_direction -= 180
        
        return np.array(self.agent_state), self.info # !! change 


    def step(self, action):
        new_state = list(self.agent_state) 

        if action == 0 and self.agent_state[0] > 0:  # Move up
            new_state[0] -= 1
            self.agent_direction = 0
        elif action == 1 and self.agent_state[0] < self.grid_size - 1:  # Move down
            new_state[0] += 1
            self.agent_direction = 180
        elif action == 2 and self.agent_state[1] > 0:  # Move left
            new_state[1] -= 1
            self.agent_direction = 90
        elif action == 3 and self.agent_state[1] < self.grid_size - 1:  # Move right
            new_state[1] += 1
            self.agent_direction = 270

        if tuple(new_state) not in self.wall_states:
            self.agent_state = tuple(new_state)

        if self.agent_state == self.in_portal:
            self.reward += 10
            self.agent_state = self.out_portal

        elif self.agent_state == self.out_portal:
            self.reward -= 30

        elif self.agent_state == self.reversal_portal_1:
            self.agent_state = (5, 2)
            self.agent_direction += 180
            self.reward -= 60

        elif self.agent_state == self.reversal_portal_2:
            self.agent_state = (7, 10)
            self.agent_direction += 180
            self.reward -= 60

        elif np.array_equal(self.agent_state, self.goal_state):
            self.reward += 100
            self.done = True

        elif any((self.agent_state == np.array(hell)).all() for hell in self.hell_states):
            self.reward = -200
            self.done = True

        else: 
            self.reward -= 0.4
            self.done = False
            # self.reset()

        self.info["Distance to Goal"] = np.sqrt((self.agent_state[0] - self.goal_state[0]) ** 2 +
                                                (self.agent_state[1] - self.goal_state[1]) ** 2)
        
        return np.array(self.agent_state), self.reward, self.done, self.info


    def render(self):
        try:
            wall_image = py.image.load('background/brown brick.png').convert_alpha()
            wall_image = py.transform.scale(wall_image, (self.cell_size, self.cell_size))
        except FileNotFoundError:
            wall_image = py.Surface((self.cell_size, self.cell_size))
            wall_image.fill((139, 69, 19))
        
        try:
            hell_image = py.image.load("background/enemy.png").convert_alpha()
            hell_image = py.transform.scale(hell_image, (self.cell_size, self.cell_size))
        except FileNotFoundError:
            hell_image = py.Surface((self.cell_size, self.cell_size))
            hell_image.fill((255, 0, 0))
        
        try:
            porter_in_image = py.image.load("background/porter_in.png").convert_alpha()
            porter_in_image = py.transform.scale(porter_in_image, (self.cell_size, self.cell_size))
        except FileNotFoundError:
            porter_in_image = py.Surface((self.cell_size, self.cell_size))
            porter_in_image.fill((0, 0, 255))
        
        try:
            porter_out_image = py.image.load("background/porter_out.png").convert_alpha()
            porter_out_image = py.transform.scale(porter_out_image, (self.cell_size, self.cell_size))
        except FileNotFoundError:
            porter_out_image = py.Surface((self.cell_size, self.cell_size))
            porter_out_image.fill((0, 255, 0))
        
        try:
            reverse_image_1 = py.image.load("background/reverse_1.png").convert_alpha()
            reverse_image_1 = py.transform.scale(reverse_image_1, (self.cell_size, self.cell_size))
        except FileNotFoundError:
            reverse_image_1 = py.Surface((self.cell_size, self.cell_size))
            reverse_image_1.fill((255, 255, 0))
        
        try:
            reverse_image_2 = py.image.load("background/reverse_2.png").convert_alpha()
            reverse_image_2 = py.transform.scale(reverse_image_2, (self.cell_size, self.cell_size))
        except FileNotFoundError:
            reverse_image_2 = py.Surface((self.cell_size, self.cell_size))
            reverse_image_2.fill((255, 255, 0))
        
        try:
            finish_image = py.image.load("background/target.jpeg").convert_alpha()
            finish_image = py.transform.scale(finish_image, (self.cell_size, self.cell_size))
        except FileNotFoundError:
            finish_image = py.Surface((self.cell_size, self.cell_size))
            finish_image.fill((0, 255, 255))

        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                grid = py.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                py.draw.rect(self.screen, (0, 191, 255), grid, 1)

        for wall_state in self.wall_states:
            self.screen.blit(wall_image, (wall_state[1] * self.cell_size, wall_state[0] * self.cell_size))

        for hell_state in self.hell_states:
            x, y = hell_state
            self.screen.blit(hell_image, (y * self.cell_size, x * self.cell_size))  
            py.draw.rect(self.screen, (250, 165, 0), (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size), 2)
  
        porter_in_image_x = self.in_portal[1] * self.cell_size
        porter_in_image_y = self.in_portal[0] * self.cell_size
        self.screen.blit(porter_in_image, (porter_in_image_x, porter_in_image_y))
        py.draw.rect(self.screen, (0, 0, 0), (porter_in_image_x, porter_in_image_y, self.cell_size, self.cell_size))
      
        porter_out_image_x = self.out_portal[1] * self.cell_size
        porter_out_image_y = self.out_portal[0] * self.cell_size
        self.screen.blit(porter_out_image, (porter_out_image_x, porter_out_image_y))
        py.draw.rect(self.screen, (0, 0, 0), (porter_out_image_x, porter_out_image_y, self.cell_size, self.cell_size))
   
        self.screen.blit(reverse_image_1, (self.reversal_portal_1[1] * self.cell_size, self.reversal_portal_1[0] * self.cell_size))
        self.screen.blit(reverse_image_2, (self.reversal_portal_2[1] * self.cell_size, self.reversal_portal_2[0] * self.cell_size))
        self.screen.blit(finish_image, (self.goal_state[1] * self.cell_size, self.goal_state[0] * self.cell_size))

        agent_rect = py.Rect(self.agent_state[1] * self.cell_size, self.agent_state[0] * self.cell_size, self.cell_size, self.cell_size)
        rotate_image = py.transform.rotate(self.agent_image, self.agent_direction)
        rotate_rect = rotate_image.get_rect(center=agent_rect.center)
        self.screen.blit(rotate_image, rotate_rect.topleft)

        py.display.flip()
        py.display.update()

def create_env(goal_coordinates,
               hell_state_coordinates):
    env = Akp_Env(goal_state=goal_coordinates, hell_state=hell_state_coordinates)
    return env
    