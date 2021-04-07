from collections import deque
import random

import numpy as np
import keras
from keras.models import load_model
from keras import Input
from keras.layers import concatenate, Dense
from keras import layers


class SwitchDQN():
    def __init__(self, state_input_size, switch_input_size,
                 gamma=0.85, learning_rate=1.e-4,
                 epsilon_decay=0.995, epsilon_min=0.01, tau=0.125,
                 batch_size=32, memory_len=2000):
        self.state_space = state_input_size
        self.switch_size = switch_input_size

        self.memory = deque(maxlen=memory_len)
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = np.ones((1, self.switch_size))
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau

        self.model = None
        self.target_model = None
        self.added_layers = list()
        self.layers = list()
        self.target_layers = list()
        self.action_space = 0
        self.compile_settings = list()

    def add(self, layer):
        if len(self.layers) == 0:
            state_input = Input(shape=(self.state_space,), name='state')
            switch_input = Input(shape=(self.switch_size,), name='switch')

            self.layers.append(switch_input)
            self.layers.append(state_input)

        if len(self.target_layers) == 0:
            state_input = Input(shape=(self.state_space,), name='state')
            switch_input = Input(shape=(self.switch_size,), name='switch')

            self.target_layers.append(switch_input)
            self.target_layers.append(state_input)

        # add new layer (should be Dense) and set it as output
        if len(self.layers) == 2:
            block_input = concatenate(
                [self.layers[-1], self.layers[0]])
            self.added_layers.append(layer)
            new_layer = layer(block_input)
            self.layers.append(new_layer)
        else:
            self.added_layers.append(layer)
            new_layer = layer(self.layers[-1])
            self.layers.append(new_layer)

        if len(self.target_layers) == 2:
            block_input = concatenate(
                [self.target_layers[-1], self.target_layers[0]])
            new_layer = layer(block_input)
            self.target_layers.append(new_layer)
        else:
            new_layer = layer(self.target_layers[-1])
            self.target_layers.append(new_layer)

    def compile(self, optimizer, loss, **kwargs):
        self.compile_settings = (optimizer, loss, kwargs)
        self.model = keras.Model(
            inputs=[self.layers[1], self.layers[0]],
            outputs=[self.layers[-1]]
        )
        self.target_model = keras.Model(
            inputs=[self.target_layers[1], self.target_layers[0]],
            outputs=[self.target_layers[-1]]
        )

        self.action_space = self.target_model.layers[-1].get_config()['units']

        self.model.compile(
            optimizer=optimizer, loss=loss, **kwargs)
        self.model.compile(
            optimizer=optimizer, loss=loss, **kwargs)

    def summary(self):
        self.model.summary()

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.added_layers = list()
        self.layers = list()
        self.target_layers = list()
        saved_model = load_model(path)

        for layer in saved_model.get_config()['layers']:
            layer_type = layer['class_name']
            if layer_type == 'InputLayer':
                if layer['name'] == 'state':
                    self.state_space = layer['config']['batch_input_shape'][-1]
                elif layer['name'] == 'switch':
                    self.switch_size = layer['config']['batch_input_shape'][-1]
            elif layer_type == 'Concatenate':
                pass
            elif layer_type == 'Dense':
                self.add(
                    Dense.from_config(layer['config'])
                )
            else:
                raise ValueError(
                    "Unsupported layer type {}".format(layer_type))

    def extend_switch(self, count=1, method='random'):
        orig_switch_size = self.switch_size
        self.switch_size += count

        state_input = Input(shape=(self.state_space,), name='state')
        switch_input = Input(shape=(self.switch_size,), name='switch')

        target_state_input = Input(shape=(self.state_space,), name='state')
        target_switch_input = Input(shape=(self.switch_size,), name='switch')

        new_layers = [switch_input, state_input]
        new_target_layers = [target_switch_input, target_state_input]

        block_input = concatenate([new_layers[-1], new_layers[0]])
        new_layer = Dense.from_config(self.added_layers[0].get_config())(block_input)
        new_layers.append(new_layer)

        target_block_input = concatenate([new_target_layers[-1], new_target_layers[0]])
        new_layer = Dense.from_config(self.added_layers[0].get_config())(target_block_input)
        new_target_layers.append(new_layer)

        for layer in self.added_layers[1:]:
            new_layer = Dense.from_config(layer.get_config())(new_layers[-1])
            new_layers.append(new_layer)

            new_layer = Dense.from_config(layer.get_config())(new_target_layers[-1])
            new_target_layers.append(new_layer)

        prev_model_weights = self.model.get_weights()
        self.layers = new_layers

        prev_target_model_weights = self.target_model.get_weights()
        self.target_layers = new_target_layers

        self.compile(self.compile_settings[0], self.compile_settings[1])

        for idx, layer in enumerate(self.model.layers[3:]):
            new_weights = layer.get_weights()
            # TODO: only works for loaded models
            prev_weights = prev_model_weights[idx * 2]

            if new_weights[0].shape == prev_weights.shape:
                new_weights[0] = prev_weights
            else:
                prev_dims = prev_weights.shape
                new_weights[0][:prev_dims[0]] = prev_weights
                if method == 'random':
                    pass
                elif method == 'mean':
                    means = prev_weights[-orig_switch_size:].mean(axis=0)
                    new_weights[0][prev_dims[0]:] = means
                elif method == 'zero':
                    new_weights[0][prev_dims[0]:] = np.zeros((1, prev_dims[1]))
                else:
                    raise ValueError("Unknown method '{}'".format(method))

            layer.set_weights(new_weights)

        for idx, layer in enumerate(self.target_model.layers[3:]):
            new_weights = layer.get_weights()
            # TODO: only works for loaded models
            prev_weights = prev_target_model_weights[idx * 2]

            if new_weights[0].shape == prev_weights.shape:
                new_weights[0] = prev_weights
            else:
                prev_dims = prev_weights.shape
                new_weights[0][:prev_dims[0]] = prev_weights
                if method == 'random':
                    pass
                elif method == 'mean':
                    means = prev_weights[-orig_switch_size:].mean(axis=0)
                    new_weights[0][prev_dims[0]:] = means
                elif method == 'zero':
                    new_weights[0][prev_dims[0]:] = np.zeros((1, prev_dims[1]))
                else:
                    raise ValueError("Unknown method '{}'".format(method))

            layer.set_weights(new_weights)

    def get_switch_size(self):
        return self.switch_size

    def act(self, state_input, switch_input):
        if self.model is None:
            raise RuntimeError("Must compile model first!")

        switch_id = np.argmax(switch_input[0])
        self.epsilon[0][switch_id] *= self.epsilon_decay
        self.epsilon[0][switch_id] = max(
            self.epsilon_min,
            self.epsilon[0][switch_id]
        )

        if np.random.random() < self.epsilon[0][switch_id]:
            action_id = np.random.randint(0, self.action_space)
        else:
            network_input = {
                'state': state_input,
                'switch': switch_input
            }
            action_id = np.argmax(self.model.predict(network_input)[0])

        return action_id

    def remember(self, state, switch, action, reward, new_state, done):
        self.memory.append([state, switch, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        for state, switch, action, reward, new_state, done in samples:
            network_input = {
                'state': state,
                'switch': switch
            }

            target = self.target_model.predict(network_input)
            if done:
                target[0][action] = reward
            else:
                new_state_network_input = {
                    'state': new_state,
                    'switch': switch
                }

                q_vals = self.target_model.predict(new_state_network_input)[0]
                next_Q = max(q_vals)

                target[0][action] = reward + self.gamma * next_Q

            self.model.fit(network_input, target, epochs=1, verbose=0)

    def target_model_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = (
                weights[i] * self.tau + target_weights[i] * (1 - self.tau)
            )
        self.target_model.set_weights(target_weights)


if __name__ == '__main__':
    test = SwitchDQN(4, 2)
    # test.add(Dense(8, use_bias=False, activation='relu'))
    test.add(Dense(4, use_bias=False, activation='relu'))

    test.compile(loss='mse', optimizer='adam')

    test_input = {
        'state': np.array([3, 3, 2, 2]).reshape((1, 4)),
        'switch': np.array([1, 0]).reshape((1, 2))
    }

    print(test.target_model.get_weights())
    print(test.target_model.predict(test_input))
    test.extend_switch(method='mean')
    print(test.target_model.get_weights())
    test_input = {
        'state': np.array([3, 3, 2, 2]).reshape((1, 4)),
        'switch': np.array([1, 0, 0]).reshape((1, 3))
    }
    print(test.target_model.predict(test_input))
