#from __future__ import annotations
import torch
import sys
import gym
import gymnasium


from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
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
dx = [0, 0, -1, 1]
dy = [1, -1, 0, 0]

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=20,
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
        self.key_pos_three = None
        self.key_picked_up = False
        self.key_picked_up_two = False
        self.key_picked_up_three = False
        self.prev_pos = None
        self.door_pos = None
        self.door_pos_two = None
        self.door_pos_three = None
        self.monster_pos_arr = []
        self.wall_arr = []
        self.size = size

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
        key_two_y = torch.randint(low=1, high=12, size=(1,)).item()

        key_three_x = torch.randint(low=1, high=3, size=(1,)).item()
        key_three_y = torch.randint(low=1, high=7, size=(1,),).item()

        door_two_x = torch.randint(low=1, high=3, size=(1,)).item()
        door_two_y = torch.randint(low=1, high=8, size=(1,)).item()
        door_three_y = torch.randint(low=1, high=9, size=(1,)).item()
        door_three_x = torch.randint(low=1, high=4, size=(1,)).item()


        get_the_x = torch.randint(low=1, high=7, size=(1,)).item()
        get_the_y = torch.randint(low=1, high=7, size=(1,)).item()

        monster_1_x = torch.randint(low=1, high=key_y + key_x, size=(1,)).item()
        monster_1_y = torch.randint(low=4, high=14, size=(1,)).item()

        monster_2_x = torch.randint(low=key_y + key_x + 1, high = key_two_x + door_two_x, size=(1,)).item()
        monster_2_y = torch.randint(low=2, high=11, size=(1,)).item()

        monster_3_x = torch.randint(low=key_y + key_x + 1, high = 19, size=(1,)).item()
        monster_3_y = torch.randint(low=1, high=door_two_y + door_three_x, size=(1,)).item()

        monster_4_x = torch.randint(low=key_y + key_x + 1, high = 19, size=(1,)).item()
        monster_4_y = torch.randint(low=door_two_y + door_three_x + 1, high=18, size=(1,)).item()

        monster_5_x = torch.randint(low = key_two_x + door_two_x + 1, high=19, size=(1,)).item()
        monster_5_y = torch.randint(low=door_two_y + door_three_y + 1, high=19, size=(1,)).item()

        self.monster_pos_arr = [(monster_1_x, monster_1_y), (monster_2_x, monster_2_y), (monster_3_x, monster_3_y), (monster_4_x, monster_4_y), (monster_5_x, monster_5_y)]

        # print(key_x)
        # print(get_the_y)
        self.grid.set(key_x, get_the_y, Key(COLOR_NAMES[0]))
        self.grid.set(key_two_x, key_two_y, Key(COLOR_NAMES[0]))
        self.grid.set(key_three_x + key_two_x + door_two_x, key_three_y, Key(COLOR_NAMES[0]))

        self.key_pos = (key_x, get_the_y)
        self.key_pos_two = (key_two_x, key_two_y)
        self.key_pos_three = (key_three_x + key_two_x + door_two_x, key_three_y)

        for i in range(0, height):
            self.grid.set(key_y+key_x, i, Wall())
            self.wall_arr.append((key_y + key_x, i))

        for i in range(0, height):
            self.grid.set(key_two_x + door_two_x, i, Wall())
            self.wall_arr.append((key_two_x + door_two_x, i))

        for i in range(key_two_x + door_two_x + 1, height):
            self.grid.set(i, door_two_y + door_three_y, Wall())
            self.wall_arr.append((i, door_two_y + door_three_y))

        self.grid.set(key_x + key_y, get_the_x, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(key_two_x + door_two_x, door_two_y, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(key_two_x + door_two_x + door_three_x, door_two_y + door_three_y, Door(COLOR_NAMES[0], is_locked=True))
        self.door_pos = (key_x + key_y, get_the_x)
        self.door_pos_two = (key_two_x + door_two_x, door_two_y)
        self.door_pos_three = (key_two_x + door_two_x + door_three_x, door_two_y + door_three_y)

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
    def grabbed_key_three(self, pos):
        if self.key_picked_up_three == True:
            return 

        if self.direction == 0 and pos[0] == self.key_pos_three[0] - 1 and pos[1] == self.key_pos_three[1]:
            self.key_picked_up_three = True
            self.grid.set(self.key_pos_three[0], self.key_pos_three[1], None)
        elif self.direction == 1 and pos[0] == self.key_pos_three[0] and pos[1] == self.key_pos_three[1] - 1:
            self.key_picked_up_three = True
            self.grid.set(self.key_pos_three[0], self.key_pos_three[1], None)
        elif self.direction == 2 and pos[0] == self.key_pos_three[0] + 1 and pos[1] == self.key_pos_three[1]:
            self.key_picked_up_three = True
            self.grid.set(self.key_pos_three[0], self.key_pos_three[1], None)
        elif self.direction == 3 and pos[0] == self.key_pos_three[0] and pos[1] == self.key_pos_three[1] + 1:
            self.key_picked_up_three = True
            self.grid.set(self.key_pos_three[0], self.key_pos_three[1], None)
        else:
            return 

        obj = self.grid.get(self.door_pos_three[0], self.door_pos_three[1])
        obj.is_locked = False
    

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
        obj.is_locked = False

    def update_ball_pos(self):
        for i in range(len(self.monster_pos_arr)):
            curr_pos = self.monster_pos_arr[i]
            #print(curr_pos)
            dir = int(torch.randint(low=0, high=4, size=(1,)).item())
            curr_pos = (curr_pos[0] + dx[dir], curr_pos[1] + dy[dir])
            if curr_pos[0] == 0 or curr_pos[0] == self.size - 1:
                continue

            if curr_pos[1] == 0 or curr_pos[1] == self.size - 1:
                continue 
            
            if curr_pos[0] == self.key_pos[0] and curr_pos[0] == self.key_pos[1]:
                continue

            if curr_pos[0] == self.key_pos_two[0] and curr_pos[0] == self.key_pos_two[1]:
                continue
                
            if curr_pos[0] == self.key_pos_three[0] and curr_pos[0] == self.key_pos_three[1]:
                continue

            lto = False
            for j in range(len(self.wall_arr)):
                if curr_pos[0] == self.wall_arr[j][0] and curr_pos[1] == self.wall_arr[j][1]:
                    lto = True
                    break
            if lto == True:
                continue 

            for j in range(i):
                if curr_pos[0] == self.monster_pos_arr[j][0] and curr_pos[1] == self.monster_pos_arr[j][1]:
                    lto = True
                    break

            if lto == True:
                continue

            self.grid.set(self.monster_pos_arr[i][0], self.monster_pos_arr[i][1], None)
            self.grid.set(curr_pos[0], curr_pos[1], Ball(COLOR_NAMES[0]))
            self.monster_pos_arr[i] = curr_pos

        return 

    def step(self, action):
        temp = super().step(action)
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
                if self.direction == 0 and self.agent_pos[0] == self.door_pos_two[0] - 1 and self.agent_pos[1] == self.door_pos_two[1]:
                    self.grid.set(self.door_pos_two[0], self.door_pos_two[1], None)

            if self.key_picked_up_three == False:
                self.grabbed_key_three(self.agent_pos)
            else:
                print(self.direction)
                print(self.door_pos_three)
                print(self.agent_pos)
                if self.direction == 1 and self.agent_pos[0] == self.door_pos_three[0] and self.agent_pos[1] == self.door_pos_three[1] - 1:
                    self.grid.set(self.door_pos_three[0], self.door_pos_three[1], None)

        # print(Actions)
        self.update_ball_pos()
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