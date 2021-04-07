from math import sqrt
import random

import numpy as np
# import cv2


class DiscreteGridworld:
    def __init__(self, height, width, goal=None, dir_count=4,
                 reward='sparse', position_representation=4,
                 graphic=False):

        if type(height) != int or type(width) != int:
            raise TypeError("World dimensions must be integers")
        if height < 1:
            raise ValueError("World height must be at least 1")
        if width < 1:
            raise ValueError("World width must be at least 1")

        self.width = width
        self.height = height

        if len(goal) is None:
            self.goal = [(
                random.randint(0, width),
                random.randint(0, height)
            )]
        else:
            if type(goal) != list:
                raise TypeError("Goal positions must be a list")
            if len(goal) != 2:
                raise ValueError("Goal position must be a list of two values")
            goal_x, goal_y = goal
            if type(goal_x) != int or type(goal_y) != int:
                raise TypeError("Goal coordinates must be integers")
            if goal_x < 0 or goal_x > self.width:
                raise ValueError(
                    "Goal x coordinate must be from interval [1, width]")
            if goal_y < 0 or goal_y > self.height:
                raise ValueError(
                    "Goal y coordinate must be from interval [1, height]")
            self.goal = goal

        if dir_count == 4:
            self.possible_dirs = ['N', 'E', 'S', 'W']
        elif dir_count == 8:
            self.possible_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        else:
            raise ValueError("Only 4 or 8 directions are supported")

        if reward == 'simple':
            self.rew_func = self.get_reward_simple
        elif reward == 'sparse':
            self.rew_func = self.get_reward_sparse
        elif reward == 'costly':
            self.rew_func = self.get_reward_costly
        elif reward == 'delta':
            self.rew_func = self.get_reward_delta
        elif reward == 'simple_dist':
            self.rew_func = self.get_diff_reward
        else:
            raise ValueError("Unknown reward function {}".format(reward))

        if position_representation != 4 and position_representation != 2:
            raise ValueError(
                "Position representation can be 2 or 4 (got {})".format(
                    position_representation
                )
            )

        self.position_representation = position_representation
        self.use_graphic = graphic

        self.current_position = [None, None]

    def __str__(self):
        return "World of size {}x{}; goal position: [{}, {}]".format(
            self.height, self.width, self.goal[0], self.goal[1])

    def get_world_size(self):
        return self.height * self.width

    def get_world_shape(self):
        return (self.height, self.width)

    def generate_image(self):
        img = np.full((self.height + 1, self.width + 1, 3), 255, dtype=np.uint8)

        # we add the goal to the G channel
        if self.position_representation == 4:
            goal_x, goal_y = self.goal
            goal_y = self.height - goal_y
            img[goal_y, goal_x, 1] = 0
            img[goal_y, goal_x, 2] = 0

        # we add the current position to the R channel
        if self.current_position != [None, None]:
            curr_x, curr_y = self.current_position
            curr_y = self.height - curr_y
            img[curr_y, curr_x, 0] = 0
            img[curr_y, curr_x, 2] = 0

        img = np.expand_dims(img, axis=0)
        return img

    def get_position(self):
        if self.use_graphic:
            return self.generate_image()

        if self.position_representation == 4:
            return np.array(
                self.current_position + self.goal
            ).reshape((1, 4))
        return np.array(self.current_position).reshape((1, 2))

    def get_goal(self):
        return self.goal

    def get_state_shape(self):
        return self.get_position().shape

    def get_action_space_size(self):
        return len(self.possible_dirs)

    def get_change(self, direction):
        if direction == 'N':
            return (0, 1)
        elif direction == 'NE':
            return (1, 1)
        elif direction == 'E':
            return (1, 0)
        elif direction == 'SE':
            return (1, -1)
        elif direction == 'S':
            return (0, -1)
        elif direction == 'SW':
            return (-1, -1)
        elif direction == 'W':
            return (-1, 0)
        elif direction == 'NW':
            return (-1, 1)
        elif direction is None:
            return (0, 0)
        else:
            raise ValueError("Unknown direction {}".format(direction))

    def get_reward_simple(self, old_pos, new_pos):
        if self.goal == new_pos:
            return 1

        if new_pos == old_pos:
            return -1

        return 0

    def get_reward_sparse(self, old_pos, new_pos):
        if self.goal == new_pos:
            return 1

        return -1

    def get_reward_costly(self, old_pos, new_pos):
        if self.goal == new_pos:
            return 100

        if new_pos == old_pos:
            return -1

        return -0.01

    def get_simple_distance(self, pos_1, pos_2):
        x_1, y_1 = pos_1
        x_2, y_2 = pos_2
        return abs(x_1 - x_2) + abs(y_1 - y_2)

    def get_diff_reward(self, old_pos, new_pos):
        if self.goal == new_pos:
            return 1

        if new_pos == old_pos:
            return -100

        old_d = self.get_simple_distance(self.goal, old_pos)
        new_d = self.get_simple_distance(self.goal, new_pos)

        if new_d < old_d:
            return 1
        else:
            return -1

    def get_distance(self, pos_1, pos_2):
        x_1, y_1 = pos_1
        x_2, y_2 = pos_2

        return sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    def get_reward_delta(self, old_pos, new_pos):
        if old_pos == new_pos:
            return -100

        if new_pos == self.goal:
            return 100

        old_d = self.get_distance(self.goal, old_pos)
        new_d = self.get_distance(self.goal, new_pos)

        return old_d - new_d

    def is_done(self):
        return self.goal == self.current_position

    def reset(self):
        self.current_position = [
            random.randint(0, self.width),
            random.randint(0, self.height)
        ]
        if self.current_position == self.goal:
            return self.reset()
        return self.get_position()

    def step(self, action_id):
        if self.current_position == [None, None]:
            self.reset()

        if action_id >= len(self.possible_dirs):
            raise ValueError(
                "Unknown action {}, world supports only {} actions".format(
                    action_id, len(self.possible_dirs)
                ))
        action = self.possible_dirs[action_id]

        c_x, c_y = self.current_position
        x_change, y_change = self.get_change(action)
        new_x = max(min(c_x + x_change, self.width), 0)
        new_y = max(min(c_y + y_change, self.height), 0)
        new_pos = [new_x, new_y]

        reward = self.rew_func(self.current_position, new_pos)
        self.current_position = new_pos
        done = self.is_done()

        info = {
            "state1": [c_x, c_y],
            "action": action,
            "state2": self.current_position,
            "reward": reward,
            "done": done,
            "string":
                "Moved from {} to {} (action {}); reward: {}, done: {}".format(
                [c_x, c_y], self.current_position, action, reward, done
            )
        }

        return self.get_position(), reward, done, info
