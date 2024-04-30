#from __future__ import annotations
import torch
import sys
import gym
import gymnasium
import pickle

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
#from minigrid.minigrid_env import MiniGrid-Empty-5x5-v0

#10 Environments
#1) Point A to Point B - # of steps c
#2) Point A to Point B - One Wall with an Opening c
#3) Point A to Point B - One Wall With A Key To Open c
#4) Point A to Point B - One Wall With Two Keys and a Door to Open c
#5) Point A to Point B - Two Walls with Two Keys to Open
#6) Point A to Point B back to Point A with One Wall and One Key
#7) Point A to Point b - Three Keys Three Walls
#8) ----
#9) ----
#10) ----

view_new_x = [(0, 1), (-1, 1), (0, -1), (-1, 1)]
view_new_y = [(-1, 1), (0, 1), (-1, 1), (0, -1)]

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

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=15,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps = None,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.direction = 0
        self.key_pos = None
        self.key_pos_two = None
        self.key_picked_up = False
        self.key_picked_up_two = False
        self.prev_pos = None
        self.door_pos = None
        self.door_pos_two = None
        self.size = size
        self.steps = 0
        self.grid_list = []
        self.wall_list = []
        self.unlocked_1 = False
        self.unlocked_2 = False

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

    def update_grid(self, action):
        curr_view = torch.zeros((7, 6))
        curr_view[0][0] = self.agent_pos[0]
        curr_view[0][1] = self.agent_pos[1]
        curr_view[0][2] = self.direction
        update_x = view_new_x[self.direction]
        update_y = view_new_y[self.direction]
        #print(self.direction)
        
        for j in range(6):
            for k in range(6):
                temp_update_x = self.agent_pos[0] + update_x[0]*int(k/2) + update_x[1]*k
                temp_update_y = self.agent_pos[1] + update_y[0]*int(j/2) + update_y[1]*j
                temp_pos = (temp_update_x, temp_update_y)


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

                if temp_pos[0] == self.key_pos_two[0] and temp_pos[1] == self.key_pos_two[1] and self.key_picked_up_two == False:
                    curr_view[j + 1][k] = 2

                if temp_pos[0] == self.width - 2 and temp_pos[1] == self.height - 2:
                    print("asdlfhasoidfhaioshfoidhs THIS IS 3")
                    curr_view[j + 1][k] = 3
                    #print(3)
                    continue 

                if temp_pos[0] == self.door_pos[0] and temp_pos[1] == self.door_pos[1] and self.unlocked_1 == False:
                    curr_view[j + 1][k] = 4
                    #print(4)
                    continue 

                if temp_pos[0] == self.door_pos_two[0] and temp_pos[1] == self.door_pos[1] and self.unlocked_2 == False:
                    curr_view[j + 1][k] = 4
                    continue

                curr_view[j + 1][k] = 0

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

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the door and key
        #door_x = torch.randint(low=0, high=5).item()
        key_x = torch.randint(low=1, high=3, size=(1,)).item()
        key_y = torch.randint(low=1, high=3, size=(1,)).item()

        key_two_x = torch.randint(low=8, high=11, size=(1,)).item()
        key_two_y = torch.randint(low=0, high=12, size=(1,)).item()

        door_two_x = torch.randint(low=1, high=3, size=(1,)).item()
        door_two_y = torch.randint(low=0, high=13, size=(1,)).item()

        get_the_x = torch.randint(low=1, high=7, size=(1,)).item()
        get_the_y = torch.randint(low=1, high=7, size=(1,)).item()

        # print(key_x)
        # print(get_the_y)
        self.grid.set(key_x, get_the_y, Key(COLOR_NAMES[0]))
        self.grid.set(key_two_x, key_two_y, Key(COLOR_NAMES[0]))

        self.key_pos = (key_x, get_the_y)
        self.key_pos_two = (key_two_x, key_two_y)

        for i in range(0, height):
            if i == get_the_x:
                continue
            self.wall_list.append((key_y + key_x, i))
            self.grid.set(key_y+key_x, i, Wall())

        for i in range(0, height):
            if i == door_two_y:
                continue
            self.wall_list.append((key_two_x + door_two_x, i))
            self.grid.set(key_two_x + door_two_x, i, Wall())

        self.grid.set(key_x + key_y, get_the_x, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(key_two_x + door_two_x, door_two_y, Door(COLOR_NAMES[0], is_locked=True))
        self.door_pos = (key_x + key_y, get_the_x)
        self.door_pos_two = (key_two_x + door_two_x, door_two_y)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

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
    #     return obs

    #0 - right
        #1 - down
        #2 - left
        #3 - up

    def grabbed_key_two(self, pos):
        if self.key_picked_up_two == True:
            return 

        if self.direction == 0 and pos[0] == self.key_pos_two[0] - 1 and pos[1] == self.key_pos_two[1]:
            self.key_picked_up_two = True
            self.grid.set(self.key_pos_two[0], self.key_pos_two[1], None)
        elif self.direction == 1 and pos[0] == self.key_pos_two[0] and pos[1] == self.key_pos_two[1] - 1:
            self.key_picked_up_two = True
            self.grid.set(self.key_pos_two[0], self.key_pos_two[1], None)
        elif self.direction == 2 and pos[0] == self.key_pos_two[0] + 1 and pos[1] == self.key_pos_two[1]:
            self.key_picked_up_two = True
            self.grid.set(self.key_pos_two[0], self.key_pos_two[1], None)
        elif self.direction == 3 and pos[0] == self.key_pos_two[0] and pos[1] == self.key_pos_two[1] + 1:
            self.key_picked_up_two = True
            self.grid.set(self.key_pos_two[0], self.key_pos_two[1], None)
        else:
            return 

        self.unlocked_2 = True
        obj = self.grid.get(self.door_pos_two[0], self.door_pos_two[1])
        obj.is_locked = False

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
        self.unlocked_1 = True
        obj.is_locked = False

    def step(self, action):
        self.update_grid(action)
        self.steps += 1
        if self.steps >= 75:
            self.grid_list.append(torch.ones((10, 9))*-10)
            with open("trajectory_10.pkl", 'wb') as f:
                pickle.dump(self.grid_list, f)
            
            self.reset()

        temp = super().step(action)

        if self.agent_pos[0] == self.width - 2 and self.agent_pos[1] == self.height - 2:
            self.grid_list.append(torch.ones((10, 9))*10)
            with open("trajectory_8.pkl", 'wb') as f:
                pickle.dump(self.grid_list, f)
        #temp = super().step(action)
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

            if self.key_picked_up_two == False:
                self.grabbed_key_two(self.agent_pos)
            else:
                print(self.direction)
                print(self.agent_pos)
                print(self.door_pos_two)
                if self.direction == 0 and self.agent_pos[0] == self.door_pos_two[0] - 1 and self.agent_pos[1] == self.door_pos_two[1]:
                    self.grid.set(self.door_pos_two[0], self.door_pos_two[1], None)

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