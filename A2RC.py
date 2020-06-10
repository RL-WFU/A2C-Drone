import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.layers import (TimeDistributed, Activation, Dense, Flatten, Input, GRU)
import numpy as np
from env import *
import argparse

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import collections
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


def get_action(model, inputs):
    output = model(inputs)
    probs = tf.nn.softmax(output)

    # 4 is num_actions
    action = np.random.choice(4, p=probs.numpy()[0])

    return action

def write_hyper_parameters(out):
    out.write("seq_length: {}".format(args.seq_length))
    out.write("\n")
    out.write("num_steps: {}".format(args.num_steps))
    out.write("\n")
    out.write("sight_dim: {}".format(args.sight_dim))
    out.write("\n")
    out.write("num_episodes: {}".format(args.num_episodes))
    out.write("\n")
    out.write("lr: {}".format(args.lr))
    out.write("\n")
    out.write("gamma: {}".format(args.gamma))
    out.write("\n")
    out.write("extra_layer: {}".format(args.extra_layer))
    out.write("\n")
    out.write("layer_val: {}".format(args.layer_val))
    out.write("\n")
    out.write("net_config: {}".format(args.net_config))
    out.write("\n")
    out.write("image: {}".format(args.image))



class PolicyEstimator_RNN:
    def __init__(self, sight_dim=2, batch_size = 1, train_length = 6, lr=0.0001, scope='RNN_model_policy'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None, self.train_length, (self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3], dtype=tf.float32, name='state')

            if args.net_config == 0:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=args.layer_val)
                self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=args.layer_val//2)

                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val//4)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2, initial_state=self.initial_state)

                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val//4])

            elif args.net_config == 1:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=args.layer_val)
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val//2)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense1, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val//2])

            elif args.net_config == 2:
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.state, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val])

            #Goes from [1, 192] to [1, 4], may need another fully connected layer
            if args.extra_layer:
                self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=args.layer_val//4)
                self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=4)
            else:
                self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)

            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, states, target, action, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state: states, self.action: action, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

class ValueEstimator_RNN:
    def __init__(self, sight_dim = 2, batch_size = 1, train_length = 6, lr=0.0001, scope='RNN_model_value'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None, self.train_length, (self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3],
                                        dtype=tf.float32, name='state')

            if args.net_config == 0:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=args.layer_val)
                self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=args.layer_val // 2)

                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val // 4)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2,
                                                            initial_state=self.initial_state)

                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val // 4])

            elif args.net_config == 1:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=args.layer_val)
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val // 2)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense1,
                                                            initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val // 2])

            elif args.net_config == 2:
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.state, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val])

            # Goes from [1, 192] to [1, 4], may need another fully connected layer

            if args.extra_layer:
                self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                num_outputs=args.layer_val // 4)
                self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=1)
            else:
                self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            self.value_estimate = tf.squeeze(self.output)

            self.target = tf.placeholder(tf.float32, name='target')

            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, states, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: states})

    def update(self, states, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: states, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def get_last_t_states(t, episode):
    states = []
    for i, transition in enumerate(episode[-t:]):
        states.append(transition.state)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1) * 3])
    return states

def get_last_t_minus_one_states(t, episode):
    states = []
    for i, transition in enumerate(episode[-t + 1:]):
        states.append(transition.state)

    states.append(episode[-1].next_state)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1) * 3])
    return states


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=0.99):


    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(0)
    episode_lengths = np.zeros(0)
    most_common_actions = []
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        env.reset()

        episode = []
        episode_rewards = np.append(episode_rewards, 0)
        episode_lengths = np.append(episode_lengths, 0)

        classified_image = env.getClassifiedDroneImage()

        done = False
        reward = 0
        actions = []

        for t in range(args.num_steps):

            # Take a step
            if t < args.seq_length:
                action = np.random.randint(0, 4)
            else:
                states = get_last_t_states(args.seq_length, episode)
                action_probs = estimator_policy.predict(state=states)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            if t == 0:
                last_action = action
            else:
                last_action = actions[-1]

            next_image, reward, done = env.step(action, last_action)

            actions.append(action)
            # Keep track of the transition
            episode.append(Transition(
                state=classified_image, action=action, reward=reward, next_state=next_image, done=done))

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            #UPDATE EVERY TIMESTEP AFTER seq_length TIMESTEPS
            if t >= args.seq_length:
                states = get_last_t_states(args.seq_length, episode)

                states1 = get_last_t_minus_one_states(args.seq_length, episode)
                value_next = estimator_value.predict(states1)

                td_target = reward + args.gamma * value_next
                td_error = td_target - estimator_value.predict(states)

                estimator_value.update(states, td_target)
                estimator_policy.update(states, td_error, action)



            if done:
                break

            classified_image = next_image

        if i_episode % 1000 == 0 and i_episode > 0:
            episodes = [i + i_episode - 1000 for i in range(1000)]
            plt.plot(episodes, episode_rewards[i_episode-1000:i_episode])
            plt.ylabel('Episode reward')
            plt.xlabel('Episode')
            plt.savefig(os.path.join(args.save_dir, 'Rewards {} to {}.png'.format(i_episode, i_episode + 1000)))
            plt.clf()


        data = collections.Counter(actions)
        most_common_actions.append(data.most_common(1))
        print("Episode {} finished. Reward: {}. Steps: {}. Most Common Action: {}".format(i_episode,
                                                                                          episode_rewards[i_episode],
                                                                                          episode_lengths[i_episode],
                                                                                          most_common_actions[
                                                                                              i_episode]))
    plt.plot(episode_rewards)
    plt.ylabel('Episode reward')
    plt.xlabel('Episode')
    plt.savefig(os.path.join(args.save_dir, 'Rewards all episodes.png'))
    plt.clf()





parser = argparse.ArgumentParser(description='Drone A2C trainer')

parser.add_argument('--seq_length', default=6, type=int, help='number of stacked images')
parser.add_argument('--num_steps', default=1000, type=int, help='episode length')
parser.add_argument('--sight_dim', default=2, type=int, help='Drone sight distance')
parser.add_argument('--num_episodes', default=5, type=int, help='number of episodes')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
parser.add_argument('--extra_layer', default=False, help='extra hidden layer between recurrent and output')
parser.add_argument('--save_dir', default='A2RC_saves', type=str, help='Save directory')
parser.add_argument('--layer_val', default=64, type=int, help='First layer hidden units (min=64')
parser.add_argument('--net_config', default=2, type=int, help='ID of network configuration')
parser.add_argument('--image', default='image3.TIF', type=str, help='Environment image file')
args = parser.parse_args()





environment = Env(args)
policy_estimator = PolicyEstimator_RNN(args.sight_dim, lr=args.lr)
value_estimator = ValueEstimator_RNN(args.sight_dim, lr=args.lr)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

outF = open(os.path.join(args.save_dir, "hyper_parameters.txt"), "a")
write_hyper_parameters(outF)
outF.close()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    actor_critic(environment, policy_estimator, value_estimator, args.num_episodes)



#environment.plot_visited()

# TO DO:
# Currently, reward is based on only the current state. Try to import only the current state,
# and possible successors

# implement gru layer

# Using RNN, we can pass the agent single frames of the environment
# Store entire episodes, and randomly draw traces of 8ish steps from a random batch of episodes
# Keeps random sampling but also retains sequences
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc
# Another possible trick - sending the last half of the gradients back for a given trace


# NOTES
# Tends to favor one action heavily after it has success with mostly that action
# Meaning, it can only tell that action x is good, not why it's good
# RNN will probably fix this issue