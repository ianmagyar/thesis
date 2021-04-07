import random

from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from feddlbqn import FedDLBQN
from feddlbqn_limit import LimitFedDLBQN
from feddlbqn_target import TargetFedDLBQN
from randfeddlbqn import RandFedDLBQN
from dlbqn import DLBQN
from switch_dqn import SwitchDQN
from pushTheBox import PushTheBoxEnv


actions4 = ['N', 'E', 'S', 'W']
TEST_ENV_HEIGHT = 2
TEST_ENV_WIDTH = 2


def evaluate_agent_policy(agent, envs):
    correct = 0
    total = 0
    errors = []

    for env_idx in range(len(envs)):
        env = envs[env_idx]
        env_height, env_width = env.get_world_shape()
        env_state_shape = env.get_state_shape()
        goal_x, goal_y = env.get_goal()
        use_switch = True

        if use_switch:
            switch_val = np.zeros((1, len(envs)))
            switch_val[0][env_idx] = 1

        for y_i in range(env_height + 1):
            for x_i in range(env_width + 1):
                if x_i == goal_x and y_i == goal_y:
                    continue
                if env_state_shape == (1, 2):
                    state = np.array(
                        [x_i, y_i]
                    ).reshape((1, 2))
                elif env_state_shape == (1, 4):
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

                env.box_position = [x_i, y_i]
                if action != env.get_human_move():
                    errors.append(
                        (env_idx, [x_i, y_i], [goal_x, goal_y], action)
                    )
                else:
                    correct += 1
                total += 1

    return correct, total, (correct / total) * 100, errors


def train_agent(agent, envs, position_representation=4,
                training_per_env=300, verbose=False, show_plot=True,
                save_models=False, log_file=None, test_run=0):
    env_height, env_width = TEST_ENV_HEIGHT, TEST_ENV_WIDTH
    env_no = len(envs)
    # envs = []
    # e_logs = []

    # for env_idx in range(env_no):
    #     goal_pos = [
    #         random.randint(0, env_width),
    #         random.randint(0, env_height)
    #     ]
    #     human = random.choice([
    #         'horizontal',
    #         'vertical',
    #         # 'stairs'
    #     ])
    #     env = PushTheBoxnv(
    #         height=env_height, width=env_width, goal=goal_pos,
    #         human_type=human, position_representation=position_representation
    #     )
    #     envs.append(env)
    #     e_logs.append([env_width, env_height, goal_pos[0], goal_pos[1]])

    # if env_logs is not None:
    #     env_logs.write("{};{}\n".format(test_run, e_logs))
    #     env_logs.flush()

    iterations = training_per_env * env_no
    iterations_length = env_height * (env_width + 2) * 2
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
    #         gamma=0.85, learning_rate=0.005, epsilon_decay=0.95
    #     )
    #     agent_name = 'DQN'
    # elif agentType == DLBQN:
    #     agent = DLBQN(
    #         world=env, switch_count=env_no, layers=[24, 48, 24],
    #         gamma=0.85, learning_rate=0.005, epsilon_decay=0.95
    #     )
    #     agent_name = 'DLBQN'
    # elif agentType == SwitchDQN:
    #     agent = SwitchDQN(
    #         world=env, switch_count=env_no, layers=[24, 48, 24],
    #         gamma=0.85, learning_rate=0.005, epsilon_decay=0.95
    #     )
    #     agent_name = 'SwitchDQN'
    # else:
    #     raise ValueError("Unsupported agent type {}".format(agentType))

    for trial in tqdm(range(iterations), position=0, leave=True):
        best_prec = 0.0
        env_idx = random.randrange(env_no)
        env = envs[env_idx]
        cur_state = env.reset()

        switch_val = np.zeros((1, env_no))
        switch_val[0][env_idx] = 1
        agent.world = env

        for step in range(iterations_length):
            action_id = agent.act(cur_state, switch_val)

            new_state, reward, done, info = env.step(action_id)

            agent.remember(
                cur_state, switch_val, action_id, reward, new_state, done
            )
            agent.replay()
            agent.target_model_train()

            if verbose:
                print(info['string'])

            cur_state = new_state
            if done:
                break

        # print("Env {}; {} steps".format(env_idx, step + 1))
        if log_file is not None or show_plot:
            iteration_steps.append(step + 1)
            correct, total, prec, errors = evaluate_agent_policy(agent, envs)
            iteration_precisions.append(prec)
            if log_file is not None:
                log_file.write("{};{};{};{};{};{}\n".format(
                    test_run, trial + 1, correct, total, prec, errors
                ))

            if prec > best_prec:
                agent.model.save(
                    'pushthebox_models\\{}_{}_world_{}_state_test{}.h5'.format(
                        agent_name, env_no, position_representation, test_run)
                )
                best_prec = prec

            if prec == 100.0:
                break

        if save_models and (trial + 1) % 100 == 0:
            agent.model.save('iteration_{}.h5'.format(trial + 1))

    if show_plot:
        plt.plot(iteration_steps, 'b')
        plt.plot(iteration_precisions, 'r')
        plt.xlabel('Iterations')
        plt.ylabel('Steps/Precision')
        plt.hlines(env_height + env_width, 0, iterations)
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


def run_test(env_no, state):
    TEST_COUNT = 500

    # env_logs = open('pushthebox2_{}_{}_envs.csv'.format(env_no, state), 'a')
    # env_logs.write('test;envs\n')
    # env_logs.flush()

    # envs = []
    # e_logs = []

    # env_height, env_width = TEST_ENV_HEIGHT, TEST_ENV_WIDTH
    # for i in range(TEST_COUNT):
    #     envs.append(list())
    #     e_logs.append(list())
    #     for env in range(env_no):
    #         goal_pos = [
    #             random.randint(0, env_width),
    #             random.randint(0, env_height)
    #         ]
    #         human = random.choice([
    #             'horizontal',
    #             'vertical',
    #             'stairs'
    #         ])
    #         env = PushTheBoxEnv(
    #             height=env_height, width=env_width, goal=goal_pos,
    #             human_type=human, position_representation=state
    #         )
    #         envs[i].append(env)
    #         e_logs[i].append(
    #             [env_width, env_height, goal_pos[0], goal_pos[1], human])

    #     if env_logs is not None:
    #         env_logs.write("{};{}\n".format(i, e_logs[i]))
    #         env_logs.flush()

    # TODO: load from file
    world_logs = open(
        'pushthebox_{}_{}_envs.csv'.format(env_no, state), 'r')
    world_data = pd.read_csv(world_logs, sep=';', header=0, converters={'worlds': eval})
    envs = list()

    for idx, row in world_data.iterrows():
        test_idx, loaded_worlds = row['test'], row['envs']
        # print("Processing test {}".format(test_idx))
        env_strings = loaded_worlds[1:-2].split('], ')

        test_worlds = []
        for test_world in env_strings:
            data = test_world[1:].split(', ')
            width = int(data[0])
            height = int(data[1])
            goal_x = int(data[2])
            goal_y = int(data[3])
            human_type = data[4][1:-1]
            env = PushTheBoxEnv(
                height, width, goal=[goal_x, goal_y],
                human_type=human_type,
                position_representation=state
            )
            test_worlds.append(env)

        envs.append(test_worlds)

    fed_batch = 4
    big_batch = env_no * fed_batch
    test_agents = [
        # 'SwitchDQN',
        # 'DLBQN',
        # 'FedDLBQN',
        # 'TargetFedDLBQN',
        'LimitFedDLBQN'
    ]

    for a_string in test_agents:
        print("TRAINING {} agent:".format(a_string))

        log = open(
            'pushthebox_{}_{}_{}.csv'.format(a_string, env_no, state), 'a')
        log.write("test;iteration;correct;total;perc;errors\n")
        log.flush()

        for i in tqdm(range(TEST_COUNT)):
            if a_string == 'SwitchDQN':
                agent = build_SwitchDQN(state, env_no, big_batch)
            elif a_string == 'DLBQN':
                agent = build_DLBQN(state, env_no, big_batch)
            elif a_string == 'FedDLBQN':
                agent = build_FedDLBQN(state, env_no, fed_batch)
            # elif a_string == 'RandFedDLBQN':
            #     agent = build_RandFedDLBQN(state, env_no, big_batch)
            elif a_string == 'TargetFedDLBQN':
                agent = build_TargetFedDLBQN(state, env_no, fed_batch)
            elif a_string == 'LimitFedDLBQN':
                agent = build_LimitFedDLBQN(state, env_no, fed_batch)
            else:
                raise ValueError("Unsupported agent type {}".format(a_string))

            print("Test run {}".format(i + 1))
            train_agent(
                agent=agent,
                envs=envs[i],
                position_representation=state,
                training_per_env=100,
                verbose=False,
                show_plot=False,
                save_models=False,
                log_file=log,
                test_run=i
            )
            log.flush()
        log.close()
    # env_logs.close()


if __name__ == '__main__':
    run_test(env_no=8, state=4)
    run_test(env_no=8, state=2)
