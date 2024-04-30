#from __future__ import annotations
import torch
import sys
import gym
import gymnasium
import numpy as np
import pickle


from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

import torch
import torch.nn as nn
import torch.nn.functional as F

class GridClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GridClassifier, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        #print(x.size())
        #print()
        # print(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], 12)  # Flatten
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

model = GridClassifier(num_classes=3)

# Load the model's parameters
model.load_state_dict(torch.load('model.pth'))
#from minigrid.minigrid_env import MiniGrid-Empty-5x5-v0

#10 Environments
#1) Point A to Point B - # of steps c
#2) Point A to Point B - One Wall with an Opening c
#3) Point A to Point B - One Wall With A Key To Open c
#4) Point A to Point B - One Wall With Two Keys and a Door to Open c
#5) Point A to Point B - Two Walls with Two Keys to Open
#6) Point A to Point b - Three Keys Three Walls 
#7) Monster moving up and down 
#8) ----
#9) ----
#10) ----


# class ActualManualControl(ManualControl):
#     def __init__(
#         self,
#         seed,
#         env
#     ):
#         self.env = env
#         super().__init__(
#             seed=42,
#             env=env
#         )

#     def keyDownCb(keyName):
#         if keyName == 'BACKSPACE':
#             super.resetEnv()
#             return

#         if keyName == 'ESCAPE':
#             sys.exit(0)

#         action = 0
#         print("asdadsf")

#         if keyName == 'LEFT':
#             action = self.env.actions.left
#         elif keyName == 'RIGHT':
#             action = self.env.actions.right
#         elif keyName == 'UP':
#             action = self.env.actions.forward

#         elif keyName == 'SPACE':
#             action = self.env.actions.toggle
#         elif keyName == 'c':
#             print("asdfasdfadsf")
#             action = self.env.actions.pickup
#         elif keyName == 'PAGE_DOWN':
#             action = self.env.actions.drop

#         elif keyName == 'RETURN':
#             action = self.env.actions.done

#         else:
#             print("unknown key %s" % keyName)
#             return

#         obs, reward, done, info = self.env.step(action)

#         print('step=%s, reward=%.2f' % (self.env.step_count, reward))

#         if done:
#             print('done!')
#             super.resetEnv()
view_arr_x = [(0, 1, 2), (-1, 0, 1), (0, -1, -2), (-1, 0, 1)]
view_arr_y = [(-1, 0, 1), (0, 1, 2), (-1, 0, 1), (0, -1, -2)]

view_new_x = [(0, 1), (-1, 1), (0, -1), (-1, 1)]
view_new_y = [(-1, 1), (0, 1), (-1, 1), (0, -1)]

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps = None,
        window_size=3,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.direction = 0
        self.window_size = window_size
        self.width = size
        self.started = False
        self.step_num = 0
        self.height = size
        self.key_pos = None
        self.key_picked_up = False
        self.size = size
        self.prev_pos = None
        self.door_pos = None
        self.actions = []
        self.grid_list = []
        self.wall_list = []
        self.curr_grid = None
        self.unlocked = False
        self.ans_list = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def new_grid(self, action):
        curr_view = torch.zeros((self.size + 1, self.size))
        curr_view[0][0] = self.agent_pos[0]
        curr_view[0][1] = self.agent_pos[1]
        curr_view[0][2] = self.direction
        print(self.agent_pos)
        for j in range(self.size):
            for k in range(self.size):
                temp_pos = (j, k)
                if temp_pos[0] <= 0 or temp_pos[0] >= self.size - 1:
                    curr_view[j + 1][k] = -1
                    #print(-1)
                    continue 
                if temp_pos[1] <= 0 or temp_pos[1] >= self.size - 1:
                    curr_view[j + 1][k] = -1
                    #print(-1)
                    continue 

                if temp_pos in self.wall_list:
                    #print("asdfasdfasdf")
                    curr_view[j + 1][k] = 1
                    #print(1)
                    continue

                # print(temp_pos)
                # print(self.key_pos)
                if temp_pos[0] == self.key_pos[0] and temp_pos[1] == self.key_pos[1] and self.key_picked_up == False:
                    print("asdpfhawepifhapwiehoaiwehfioawehf")
                    curr_view[j + 1][k] = 2
                    #print(2)
                    continue 

                if temp_pos[0] == self.width - 2 and temp_pos[1] == self.height - 2:
                    print("asdlfhasoidfhaioshfoidhs THIS IS 3")
                    curr_view[j + 1][k] = 3
                    #print(3)
                    continue 

                if temp_pos[0] == self.door_pos[0] and temp_pos[1] == self.door_pos[1] and self.unlocked == False:
                    curr_view[j + 1][k] = 4
                    #print(4)
                    continue 
                
                if temp_pos[0] == self.agent_pos[0] and temp_pos[1] == self.agent_pos[1]:
                    curr_view[j + 1][k] = 5
                    continue

                curr_view[j + 1][k] = 0

        if action == self.actions.forward:
            temp_action = 1
        elif action == self.actions.right:
            temp_action = 2
        else:
            temp_action = 3

        #print(curr_view)
        self.ans_list.append([curr_view.T, temp_action])
        #print(self.wall_list)
        return

    def update_grid(self, action):
        curr_view = torch.zeros((10, 9))
        curr_view[0][0] = self.agent_pos[0]
        curr_view[0][1] = self.agent_pos[1]
        curr_view[0][2] = self.direction
        update_x = view_new_x[self.direction]
        update_y = view_new_y[self.direction]
        #print(self.direction)

        for j in range(9):
            for k in range(9):
                temp_update_x = self.agent_pos[0] + update_x[0]*4 + update_x[1]*k
                temp_update_y = self.agent_pos[1] + update_y[0]*4 + update_y[1]*j
                temp_pos = (temp_update_x, temp_update_y)
                #print(temp_pos)
                # print(j)
                # print(k)
                # print("------")
                if temp_pos[0] <= 0 or temp_pos[0] >= self.size - 1:
                    curr_view[j + 1][k] = -1
                    #print(-1)
                    continue 
                if temp_pos[1] <= 0 or temp_pos[1] >= self.size - 1:
                    curr_view[j + 1][k] = -1
                    #print(-1)
                    continue 

                if temp_pos in self.wall_list:
                    #print("asdfasdfasdf")
                    curr_view[j + 1][k] = 1
                    #print(1)
                    continue

                # print(temp_pos)
                # print(self.key_pos)
                if temp_pos[0] == self.key_pos[0] and temp_pos[1] == self.key_pos[1] and self.key_picked_up == False:
                    print("asdpfhawepifhapwiehoaiwehfioawehf")
                    curr_view[j + 1][k] = 2
                    #print(2)
                    continue 

                if temp_pos[0] == self.width - 2 and temp_pos[1] == self.height - 2:
                    print("asdlfhasoidfhaioshfoidhs THIS IS 3")
                    curr_view[j + 1][k] = 3
                    #print(3)
                    continue 

                if temp_pos[0] == self.door_pos[0] and temp_pos[1] == self.door_pos[1] and self.unlocked == False:
                    curr_view[j + 1][k] = 4
                    #print(4)
                    continue 

                curr_view[j + 1][k] = 0 
                #print(0)

        if action == self.actions.forward:
            temp_action = 1
        elif action == self.actions.right:
            temp_action = 2
        else:
            temp_action = 3

        #print(curr_view)
        self.grid_list.append({curr_view, temp_action})
        #print(self.wall_list)
        return

    #0 - None, #1 - wall, 2 - key, 3 - goal, 4 - door
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        self.curr_grid = np.zeros((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for i in range(width):
            for j in range(height):
                if i == 0 or j == 0:
                    self.curr_grid[i][j] = 1
                elif i == width - 1 or j == height - 1:
                    self.curr_grid[i][j] = 1

        # Place the door and key
        #door_x = torch.randint(low=0, high=5).item()
        key_x = torch.randint(low=1, high=5, size=(1,)).item()
        key_y = torch.randint(low=1, high=3, size=(1,)).item()

        get_the_x = torch.randint(low=1, high=7, size=(1,)).item()
        get_the_y = torch.randint(low=1, high=7, size=(1,)).item()

        # print(key_x)
        # print(get_the_y)
        self.grid.set(key_x, get_the_y, Key(COLOR_NAMES[0]))
        self.curr_grid[key_x][get_the_y] = 2

        self.key_pos = (key_x, get_the_y)

        for i in range(0, height):
            self.grid.set(key_y+key_x, i, Wall())
            self.wall_list.append((key_y + key_x, i))

        self.grid.set(key_x + key_y, get_the_x, Door(COLOR_NAMES[0], is_locked=True))
        self.door_pos = (key_x + key_y, get_the_x)
        self.curr_grid[key_x + key_y][get_the_x] = 3

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.curr_grid[width - 2][height - 2] = 4

        #print(self.curr_grid)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    # def reset(self):
    #     obs = super().reset()
    #     self.agent_pos = self.agent_start_pos
    #     # self.put_obj(Goal(), self.width - 2, self.height - 2)
    #     return obs

    #0 - right
        #1 - down
        #2 - left
        #3 - up
    def grabbed_key(self, pos):
        if self.key_picked_up == True:
            return 
        if self.direction == 0 and pos[0] == self.key_pos[0] - 1 and pos[1] == self.key_pos[1]:
            self.key_picked_up = True
            self.grid.set(self.key_pos[0], self.key_pos[1], None)
        elif self.direction == 1 and pos[0] == self.key_pos[0] and pos[1] == self.key_pos[1] - 1:
            self.key_picked_up = True
            self.grid.set(self.key_pos[0], self.key_pos[1], None)
        elif self.direction == 2 and pos[0] == self.key_pos[0] + 1 and pos[1] == self.key_pos[1]:
            self.key_picked_up = True
            self.grid.set(self.key_pos[0], self.key_pos[1], None)
        elif self.direction == 3 and pos[0] == self.key_pos[0] and pos[1] == self.key_pos[1] + 1:
            self.key_picked_up = True
            self.grid.set(self.key_pos[0], self.key_pos[1], None)
        else:
            return 

        obj = self.grid.get(self.door_pos[0], self.door_pos[1])
        self.curr_grid[self.door_pos[0]][self.door_pos[1]] = 0
        obj.is_locked = False

    def step(self, action):
        # print(len(self.grid_list))
        #print(self.grid_list[-1])
        if self.step_num > 50:
            self.ans_list.append(torch.ones((10, 9))*-10)
            with open("trajectory_10_new.pkl", 'wb') as f:
                pickle.dump(self.ans_list, f)
            
            self.reset()

        self.step_num += 1
        #print(self.step_num)
        # print("*******************")
        #fofrward, right, left
        # print("(**(*((*(****")
        # temp_grid = list(self.grid_list[-1])[0]
        # temp_grid = temp_grid.reshape(1, temp_grid.shape[0], temp_grid.shape[1])
        # #print(model(temp_grid))
        # probs = model(temp_grid).detach().numpy()[0]
        # print(probs)

        # arg_probs = np.random.choice(len(probs), p=probs)
        # print("aosidhaiowefhaoiwehfoaiwehf")
        # if arg_probs == 0:
        #     action = self.actions.forward
        # elif arg_probs == 1:
        #     action = self.actions.right
        # else:
        #     action = self.actions.left
        self.new_grid(action)

        temp = super().step(action)
        print(self.ans_list[-1])
        if self.agent_pos[0] == self.width - 2 and self.agent_pos[1] == self.height - 2:
            self.ans_list.append(torch.ones((10, 9))*10)
            with open("trajectory_10.pkl", 'wb') as f:
                pickle.dump(self.ans_list, f)
        # print("--------")
        # print(action)
        # print(self.direction)
        # #print(self.grid.grid)
        # print("---------")
        #self.actions.append(action)
        #self.grid_list.append(self.grid)
        if action == self.actions.left:
            self.direction -= 1
        elif action == self.actions.right:
            self.direction += 1

        if self.direction > 3:
            self.direction -= 4
        elif self.direction < 0:
            self.direction += 4


        # print(action)
        # print(self.agent_pos)
        # print(self.direction)
        if action == self.actions.forward and self.prev_pos != None and self.prev_pos == self.agent_pos:
            if self.key_picked_up == False:
                self.grabbed_key(self.agent_pos)
            else:
                if self.direction == 0 and self.agent_pos[0] == self.door_pos[0] - 1 and self.agent_pos[1] == self.door_pos[1]:
                    self.grid.set(self.door_pos[0], self.door_pos[1], None)
                    self.unlocked = True

        # print(Actions)
        # print(len(temp))
        # print(temp[0])

        # print(temp[1])
        # print(temp[2])
        # print(temp[3])
        # print(temp[4])
        obs, reward, done, info, _ = temp
        # obs = temp[0]
        self.prev_pos = self.agent_pos
        # reward = temp[1]
        # done = temp[2]
        # info = temp[3]
        # lto = temp[4]

        if action == self.actions.pickup:
            # Check if the agent is adjacent to the key
            print(self.agent_pos)
            print("------")
            if self.agent_pos == self.key_pos:
                self.carrying = self.carrying | 1  # Set bit for carrying key
                self.remove_obj(*self.key_pos)
                self.key_pos = None

        #self.update_grid(action)

        return obs, reward, done, info, _


def main():
    env = SimpleEnv(render_mode="human")
    	
    #env = gymnasium.make("MiniGrid-BlockedUnlockPickup-v0")
    #env.reset()

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()