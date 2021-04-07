import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class RandomScheduler():
    def __init__(self, class_no):
        pass

    def generate(self, performance, batch_size):
        item_count = performance.shape[0]
        counts = [0] * item_count
        while sum(counts) < batch_size:
            idx = np.random.randint(0, item_count)
            counts[idx] += 1

        return counts


class AdaptiveScheduler():
    def __init__(self, class_no):
        self.class_no = class_no
        self.gamma = 0.99
        self.tau = 0.005

        self.cm_input = 1

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.build_buffer()

    def get_actor(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.class_no, self.class_no, 1))
        conv = layers.Conv2D(8, (2, 2), activation='sigmoid')(inputs)
        pool = layers.MaxPooling2D((2, 2))(conv)
        flatten = layers.Flatten()(pool)
        out = layers.Dense(24, activation='relu')(flatten)
        out = layers.Dense(24, activation='relu')(out)
        outputs = layers.Dense(self.class_no, activation='softmax', kernel_initializer=last_init)(out)

        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        state_input = layers.Input(shape=(self.class_no, self.class_no, 1))
        conv = layers.Conv2D(8, (2, 2), activation='sigmoid')(state_input)
        pool = layers.MaxPooling2D((2, 2))(conv)
        flatten = layers.Flatten()(pool)
        state_out = layers.Dense(16, activation='relu')(flatten)
        state_out = layers.Dense(32, activation='relu')(state_out)

        action_input = layers.Input(shape=(self.class_no))
        action_out = layers.Dense(32, activation='relu')(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(24, activation='relu')(concat)
        out = layers.Dense(24, activation='relu')(out)
        outputs = layers.Dense(1, activation='sigmoid')(out)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def policy(self, state, noise_object=None):
        actor_output = self.actor_model(state)
        return [tf.squeeze(actor_output).numpy()]

    def build_buffer(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.class_no, self.class_no))
        self.action_buffer = np.zeros((self.buffer_capacity, self.class_no))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.class_no, self.class_no))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def generate(self, performance, batch_size):
        action = self.policy(np.expand_dims(performance, 0))
        dist = action[0]

        counts = np.around(batch_size * dist).astype('int')
        # print(counts)
        while np.sum(counts) > batch_size:
            counts[np.random.randint(0, counts.shape[0])] -= 1
        while np.sum(counts) != batch_size:
            counts[np.random.randint(0, counts.shape[0])] += 1

        return action, counts.tolist()

    def process(self, performance, action, reward, new_performance):
        self.record((performance, action, reward, new_performance))

        self.learn()

        self.update_target(
            self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(
            self.target_critic.variables, self.critic_model.variables, self.tau)


if __name__ == '__main__':
    test = AdaptiveScheduler(10)
    tf.keras.utils.plot_model(test.actor_model, 'actor.png', show_shapes=True, dpi=200)
    tf.keras.utils.plot_model(test.critic_model, 'critic.png', show_shapes=True, dpi=200)