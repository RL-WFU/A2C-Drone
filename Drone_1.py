import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (TimeDistributed, Activation, Dense, Flatten, Input, GRU)
import numpy as np
from env import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import collections
import matplotlib.pyplot as plt

def build_model(input_shape, num_actions, batch_size):
    inputs = Input(shape=input_shape)

    inputs_flat = Flatten()(inputs)

    hidden = (Dense(256, activation='relu'))(inputs_flat)

    hidden = (Dense(64, activation='relu'))(hidden)

    hidden = tf.reshape(hidden, (batch_size, batch_size, 64))

    gru = GRU(32)(hidden)

    output = Dense(num_actions)(gru)

    model = keras.Model(inputs=inputs, outputs=output, name='Please_work')

    model.summary()

    return model

def get_action(model, inputs):

    output = model(inputs)
    probs = tf.nn.softmax(output)

    #4 is num_actions
    action = np.random.choice(4, p=probs.numpy()[0])

    return action

class PolicyEstimator_GRU:
    def __init__(self, num_actions, batch_size = 1, lr = 0.00001, scope='policy_estimator'):
        self.num_actions = num_actions
        self.batch_size = batch_size
        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, shape=(6, 6, 3), name='state')
            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.state_expanded = tf.expand_dims(self.state, 0)
            self.state_flat = tf.contrib.layers.flatten(inputs=self.state_expanded)
            self.dense1 = tf.contrib.layers.fully_connected(inputs = self.state_flat, num_outputs=128)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=32)

            #HERE WE WILL EVENTUALLY PUT A 32 LAYER GRU NETWORK

            self.output = tf.contrib.layers.fully_connected(inputs=self.dense2, num_outputs=self.num_actions, activation_fn=None)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state : state, self.action : action, self.target : target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

class ValueEstimator_GRU:
    def __init__(self, lr=0.00001, scope='value_estimator'):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=(6, 6, 3), name='state')
            self.target = tf.placeholder(dtype=tf.float32, name='target')

            self.state_expanded = tf.expand_dims(self.state, 0)
            self.state_flat = tf.contrib.layers.flatten(inputs=self.state_expanded)
            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state_flat, num_outputs=128)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=32)

            # HERE WE WILL EVENTUALLY PUT A 32 LAYER GRU NETWORK

            self.output = tf.contrib.layers.fully_connected(inputs=self.dense2, num_outputs=1,
                                                            activation_fn=None)

            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state : state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state : state, self.target : target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class PolicyEstimator:
    def __init__(self, num_actions, batch_size = 1, lr = 0.0001, scope='policy_estimator'):
        self.num_actions = num_actions
        self.batch_size = batch_size
        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, shape=(6, 6, 3), name='state')
            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.state_expanded = tf.expand_dims(self.state, 0)
            self.state_flat = tf.contrib.layers.flatten(inputs=self.state_expanded)
            self.dense1 = tf.contrib.layers.fully_connected(inputs = self.state_flat, num_outputs=128)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=32)

            #HERE WE WILL EVENTUALLY PUT A 32 LAYER GRU NETWORK

            self.output = tf.contrib.layers.fully_connected(inputs=self.dense2, num_outputs=self.num_actions, activation_fn=None)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state : state, self.action : action, self.target : target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

class ValueEstimator:
    def __init__(self, lr=0.0001, scope='value_estimator'):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=(6, 6, 3), name='state')
            self.target = tf.placeholder(dtype=tf.float32, name='target')

            self.state_expanded = tf.expand_dims(self.state, 0)
            self.state_flat = tf.contrib.layers.flatten(inputs=self.state_expanded)
            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state_flat, num_outputs=128)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=32)

            # HERE WE WILL EVENTUALLY PUT A 32 LAYER GRU NETWORK

            self.output = tf.contrib.layers.fully_connected(inputs=self.dense2, num_outputs=1,
                                                            activation_fn=None)

            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state : state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state : state, self.target : target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

#num_samples = 1
#model1 = build_model((6, 6, 3), 4, num_samples)
def update_1(episode, next_image, value_estimator):

    # Calculate TD Target
    value_next = estimator_value.predict(next_image)
    td_target = reward + discount_factor * value_next
    td_error = td_target - estimator_value.predict(classified_image)

    # Update the value estimator
    estimator_value.update(classified_image, td_target)

    # Update the policy estimator
    # using the td error as our advantage estimate
    estimator_policy.update(classified_image, td_error, action)

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=0.99):

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = []
    episode_lengths = []
    most_common_actions = []
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        env.reset()

        episode = []
        episode_rewards.append(0)
        episode_lengths.append(0)

        classified_image = env.getClassifiedDroneImage()

        done = False
        reward = 0
        actions = []


        for t in range(200):

            # Take a step
            action_probs = estimator_policy.predict(classified_image)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            if t == 0:
                last_action = action
            else:
                last_action = actions[-1]
            next_image, next_row, next_col, reward, done = env.step(action, last_action)

            actions.append(action)
            # Keep track of the transition
            episode.append(Transition(
                state=classified_image, action=action, reward=reward, next_state=next_image, done=done))

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            # Calculate TD Target
            value_next = estimator_value.predict(next_image)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(classified_image)

            # Update the value estimator
            estimator_value.update(classified_image, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(classified_image, td_error, action)

            if done:
                break

            classified_image = next_image

        data = collections.Counter(actions)
        most_common_actions.append(data.most_common(1))
        print("Episode {} finished. Reward: {}. Steps: {}. Most Common Action: {}".format(i_episode, episode_rewards[i_episode], episode_lengths[i_episode], most_common_actions[i_episode]))
    plt.plot(episode_rewards)
    plt.ylabel('Episode reward')
    plt.xlabel('Episode')
    plt.show()
    plt.clf()

environment = Env()

policy_estimator = PolicyEstimator(4)
value_estimator = ValueEstimator()


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    actor_critic(environment, policy_estimator, value_estimator, 10000)

environment.plot_visited()

