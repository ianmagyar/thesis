import random

from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from discrete_gridworld import DiscreteGridworld
from feddlbqn import FedDLBQN
from randfeddlbqn import RandFedDLBQN
from feddlbqn_limit import LimitFedDLBQN
from feddlbqn_target import TargetFedDLBQN
from dlbqn import DLBQN
from switch_dqn import SwitchDQN


actions4 = ['N', 'E', 'S', 'W']
TEST_WORLD_HEIGHT = 2
TEST_WORLD_WIDTH = 2


def get_good_actions(x, y, goal_x, goal_y):
    actions = list()

    if x < goal_x:
        actions.append('E')
    if x > goal_x:
        actions.append('W')
    if y < goal_y:
        actions.append('N')
    if y > goal_y:
        actions.append('S')

    return actions


def evaluate_agent_policy(agent, worlds):
    correct = 0
    total = 0
    errors = []

    for world_idx in range(len(worlds)):
        world = worlds[world_idx]
        world_height, world_width = world.get_world_shape()
        world_state_shape = world.get_state_shape()
        goal_x, goal_y = world.get_goal()
        use_switch = True

        if use_switch:
            switch_val = np.zeros((1, len(worlds)))
            switch_val[0][world_idx] = 1

        for y_i in range(world_height + 1):
            for x_i in range(world_width + 1):
                if x_i == goal_x and y_i == goal_y:
                    continue
                if world_state_shape == (1, 2):
                    state = np.array(
                        [x_i, y_i]
                    ).reshape((1, 2))
                elif world_state_shape == (1, 4):
                    state = np.array(
                        [x_i, y_i, goal_x, goal_y]
                    ).reshape((1, 4))
                else:
                    # TODO: support graphic testing
                    return
                if use_switch:
                    test_input = {
                        'state': state,
                        'switch': switch_val
                    }
                else:
                    test_input = state

                result = agent.model.predict(test_input)
                action = np.argmax(result[0])
                action = actions4[action]
                if action not in get_good_actions(x_i, y_i, goal_x, goal_y):
                    errors.append(
                        (world_idx, [x_i, y_i], [goal_x, goal_y], action)
                    )
                else:
                    correct += 1
                total += 1

    return correct, total, (correct / total) * 100, errors


def train_agent(
        agent, worlds, world_representation=4, world_graphic=False,
        training_per_world=300, verbose=False, show_plot=True,
        save_models=False, log_file=None, test_run=0):
    world_height, world_width = TEST_WORLD_HEIGHT, TEST_WORLD_WIDTH
    world_no = len(worlds)
    # worlds = []
    # w_logs = []

    # for world in range(world_no):
    #     goal_pos = [
    #         random.randint(0, world_width),
    #         random.randint(0, world_height)
    #     ]
    #     env = DiscreteGridworld(
    #         world_height, world_width, goal=goal_pos,
    #         position_representation=world_representation,
    #         graphic=world_graphic
    #     )
    #     worlds.append(env)
    #     w_logs.append([world_width, world_height, goal_pos[0], goal_pos[1]])

    # if world_logs is not None:
    #     world_logs.write("{};{}\n".format(test_run, w_logs))

    iterations = training_per_world * world_no
    iteration_length = world_height * (world_width + 2) * 2
    iteration_steps = []
    iteration_precisions = []

    if isinstance(agent, DLBQN):
        agent_name = 'DLBQN'
    elif isinstance(agent, FedDLBQN):
        agent_name = 'FedDLBQN'
    elif isinstance(agent, SwitchDQN):
        agent_name = 'SwitchDQN'
    elif isinstance(agent, RandFedDLBQN):
        agent_name = 'RandFedDLBQN'
    elif isinstance(agent, TargetFedDLBQN):
        agent_name = 'TargetFedDLBQN'
    elif isinstance(agent, LimitFedDLBQN):
        agent_name = 'LimitFedDLBQN'
    else:
        raise ValueError("Unsupported agent type {}".format(type(agent)))

    # if agentType == DQN:
    #     agent = DQN(
    #         world=env, layers=[24, 48, 24],
    #         gamma=0.9, learning_rate=0.005, epsilon_decay=0.95
    #     )
    #     agent_name = 'DQN'
    # if agentType == DLBQN:
    #     agent = DLBQN(
    #         state_input_size=world_representation, switch_input_size=world_no,
    #         gamma=0.9, learning_rate=0.005, epsilon_decay=0.95,
    #         batch_size=4
    #     )
    #     agent_name = 'DLBQN'
    # elif agentType == SwitchDQN:
    #     agent = SwitchDQN(
    #         state_input_size=world_representation, switch_input_size=world_no,
    #         gamma=0.9, learning_rate=0.005, epsilon_decay=0.95,
    #         batch_size=4
    #     )
    #     agent_name = 'SwitchDQN'
    # else:
    #     raise ValueError("Unsupported agent type {}".format(agentType))
    # agent.add(Dense(24, activation='relu', use_bias=False))
    # # agent.add(Dense(48, activation='relu', use_bias=False))
    # # agent.add(Dense(24, activation='relu', use_bias=False))
    # agent.add(Dense(4, activation='linear', use_bias=False))
    # agent.compile(loss='mse', optimizer=Adam(lr=0.005))

    for trial in tqdm(range(iterations), position=0, leave=True):
        best_prec = 0.0
        world_idx = random.randrange(world_no)
        env = worlds[world_idx]
        cur_state = env.reset()

        # if agentType != DQN:
        switch_val = np.zeros((1, world_no))
        switch_val[0][world_idx] = 1
        agent.world = env

        for step in range(iteration_length):
            # if agentType == DQN:
            #     action = agent.act(cur_state)
            # else:
            action = agent.act(cur_state, switch_val)

            new_state, reward, done, info = env.step(action)

            # if agentType == DQN:
            #     agent.remember(
            #         cur_state, action, reward, new_state, done
            #     )
            # elif agentType == DLBQN or agentType == SwitchDQN:
            agent.remember(
                cur_state, switch_val, action, reward, new_state, done
            )

            agent.replay()
            agent.target_model_train()

            if verbose:
                print(info['string'])

            cur_state = new_state
            if done:
                break

        # print("world {}; {} steps".format(world_idx, step + 1))
        if log_file is not None or show_plot:
            iteration_steps.append(step + 1)
            correct, total, prec, errors = evaluate_agent_policy(agent, worlds)
            iteration_precisions.append(prec)
            if log_file is not None:
                log_file.write("{};{};{};{};{};{}\n".format(
                    test_run, trial + 1, correct, total, prec, errors
                ))

            if prec > best_prec:
                agent.model.save(
                    'grid_models\\{}_{}_world_{}_state_test{}.h5'.format(
                        agent_name, world_no, world_representation, test_run)
                )
                best_prec = prec
            if prec == 100.0:
                break
        if save_models and (trial + 1) % 100 == 0:
            agent.model.save("iteration_{}.h5".format(trial + 1))

    if show_plot:
        plt.plot(iteration_steps, 'b')
        plt.plot(iteration_precisions, 'r')
        plt.xlabel("Iterations")
        plt.ylabel("Steps")
        plt.hlines(world_height + world_width, 0, iterations)
        plt.show()


GAMMA = 0.9
LEARNING_RATE = 0.005
EPSILON_DECAY = 0.95


def build_SwitchDQN(world_representation, world_no, batch_size):
    agent = SwitchDQN(
        state_input_size=world_representation, switch_input_size=world_no,
        gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon_decay=EPSILON_DECAY,
        batch_size=batch_size
    )
    agent.add(Dense(24, activation='relu'))
    agent.add(Dense(48, activation='relu'))
    agent.add(Dense(24, activation='relu'))
    agent.add(Dense(4, activation='linear'))
    agent.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    return agent


def build_DLBQN(world_representation, world_no, batch_size):
    agent = DLBQN(
        state_input_size=world_representation, switch_input_size=world_no,
        gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon_decay=EPSILON_DECAY,
        batch_size=batch_size
    )
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(48, activation='relu', use_bias=False))
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(4, activation='linear', use_bias=False))
    agent.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    return agent


def build_FedDLBQN(world_representation, world_no, batch_size):
    agent = FedDLBQN(
        state_input_size=world_representation, switch_input_size=world_no,
        gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon_decay=EPSILON_DECAY,
        batch_size=batch_size
    )
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(48, activation='relu', use_bias=False))
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(4, activation='linear', use_bias=False))
    agent.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    return agent


def build_RandFedDLBQN(world_representation, world_no, batch_size):
    agent = RandFedDLBQN(
        state_input_size=world_representation, switch_input_size=world_no,
        gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon_decay=EPSILON_DECAY,
        batch_size=batch_size)
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(48, activation='relu', use_bias=False))
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(4, activation='linear', use_bias=False))
    agent.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    return agent


def build_TargetFedDLBQN(world_representation, world_no, batch_size):
    agent = TargetFedDLBQN(
        state_input_size=world_representation, switch_input_size=world_no,
        gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon_decay=EPSILON_DECAY,
        batch_size=batch_size)
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(48, activation='relu', use_bias=False))
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(4, activation='linear', use_bias=False))
    agent.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    return agent


def build_LimitFedDLBQN(world_representation, world_no, batch_size):
    agent = LimitFedDLBQN(
        state_input_size=world_representation, switch_input_size=world_no,
        gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon_decay=EPSILON_DECAY,
        batch_size=batch_size)
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(48, activation='relu', use_bias=False))
    agent.add(Dense(24, activation='relu', use_bias=False))
    agent.add(Dense(4, activation='linear', use_bias=False))
    agent.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    return agent


def run_test(world_no, state):
    TEST_COUNT = 500

    # world_logs = open(
    #     'gridworld_{}_{}_worlds.csv'.format(world_no, state), 'a')
    # world_logs.write('test;worlds\n')
    # world_logs.flush()

    # worlds = []
    # w_logs = []

    # world_height, world_width = TEST_WORLD_HEIGHT, TEST_WORLD_WIDTH
    # for i in range(TEST_COUNT):
    #     worlds.append(list())
    #     w_logs.append(list())
    #     for world in range(world_no):
    #         goal_pos = [
    #             random.randint(0, world_width),
    #             random.randint(0, world_height)
    #         ]
    #         env = DiscreteGridworld(
    #             world_height, world_width, goal=goal_pos,
    #             position_representation=state,
    #             graphic=False
    #         )
    #         worlds[i].append(env)
    #         w_logs[i].append(
    #             [world_width, world_height, goal_pos[0], goal_pos[1]])

    #     if world_logs is not None:
    #         world_logs.write("{};{}\n".format(i, w_logs[i]))
    #         world_logs.flush()

    world_logs = open(
        'gridworld_{}_{}_worlds.csv'.format(world_no, state), 'r')
    world_data = pd.read_csv(world_logs, sep=';', header=0, converters={'worlds': eval})
    worlds = list()

    for idx, row in world_data.iterrows():
        test_idx, loaded_worlds = row['test'], row['worlds']
        # print("Processing test {}".format(test_idx))

        test_worlds = []
        for test_world in loaded_worlds:
            width, height, goal_x, goal_y = test_world
            env = DiscreteGridworld(
                height, width, goal=[goal_x, goal_y],
                position_representation=state
            )
            test_worlds.append(env)

        worlds.append(test_worlds)

    fed_batch = 4
    big_batch = fed_batch * world_no

    test_agents = [
        # 'SwitchDQN',
        # 'DLBQN',
        # 'FedDLBQN',
        # 'RandFedDLBQN',
        # 'TargetFedDLBQN',
        'LimitFedDLBQN'
    ]

    for a_string in test_agents:
        print("TRAINING {} agent:".format(a_string))

        log = open(
            'gridworld_{}_{}_{}.csv'.format(a_string, world_no, state), 'a')
        log.write('test;iteration;correct;total;perc;errors\n')
        log.flush()

        for i in tqdm(range(TEST_COUNT)):
            if a_string == 'SwitchDQN':
                agent = build_SwitchDQN(state, world_no, big_batch)
            elif a_string == 'DLBQN':
                agent = build_DLBQN(state, world_no, big_batch)
            elif a_string == 'FedDLBQN':
                agent = build_FedDLBQN(state, world_no, fed_batch)
            elif a_string == 'RandFedDLBQN':
                agent = build_RandFedDLBQN(state, world_no, big_batch)
            elif a_string == 'TargetFedDLBQN':
                agent = build_TargetFedDLBQN(state, world_no, fed_batch)
            elif a_string == 'LimitFedDLBQN':
                agent = build_LimitFedDLBQN(state, world_no, fed_batch)
            else:
                raise ValueError("Unsupported agent type {}".format(a_string))
            print("Test run {}".format(i + 1))
            train_agent(
                agent=agent,
                worlds=worlds[i],
                world_representation=state,
                world_graphic=False,
                training_per_world=100,
                verbose=False,
                show_plot=False,
                save_models=False,
                log_file=log,
                test_run=i
            )
            log.flush()
        log.close()
    world_logs.close()


if __name__ == '__main__':
    run_test(world_no=8, state=4)
    run_test(world_no=8, state=2)
    # run_test(world_no=1, state=4)
