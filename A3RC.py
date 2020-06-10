import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (TimeDistributed, Activation, Dense, Flatten, Input, GRU)
import numpy as np
from env import *
import argparse

import threading
import multiprocessing
from queue import Queue

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

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           num_steps):
  """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    num_steps: The number of steps the episode took to complete
  """
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"
  )
  result_queue.put(global_ep_reward)
  return global_ep_reward


class PolicyEstimator_RNN:
    def __init__(self, sight_dim=2, batch_size = 1, train_length = 6, lr=0.0001, scope='RNN_model_policy'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None, self.train_length, (self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3], dtype=tf.float32, name='state')

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
                                                                num_outputs=args.layer_val//4)
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


class MasterAgent:
    def __init__(self, env, sess=None):
        self.sess = sess or tf.get_default_session()
        self.env = env
        self.global_policy = PolicyEstimator_RNN(args.sight_dim, scope='Global_policy')
        self.global_value = ValueEstimator_RNN(args.sight_dim, scope='Global_value')
        self.save_dir = args.save_dir


    def train(self):
        res_queue = Queue()

        workers = [Worker(self.global_policy, self.global_value, res_queue, i, self.env, save_dir=self.save_dir, sess=self.sess) for i in range(args.num_workers)]

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i, worker in enumerate(workers):
            print("Starting worker{}".format(i))
            worker.start()

        moving_average_rewards = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break

        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')

        plt.savefig(os.path.join(self.save_dir, 'A3RC Moving Average.png'))




class Worker(threading.Thread):
    global_episode = 0

    global_moving_average_reward = 0

    best_score = 0
    save_lock = threading.Lock()

    episode_rewards = []
    episode_lengths = []

    def __init__(self, global_policy, global_value, result_queue, idx, env, save_dir, sess):
        super(Worker, self).__init__()

        self.global_policy = global_policy
        self.global_value = global_value

        self.result_queue = result_queue

        self.local_policy = PolicyEstimator_RNN(args.sight_dim, lr=args.lr, scope='Local_policy{}'.format(idx))
        self.local_value = ValueEstimator_RNN(args.sight_dim, lr=args.lr, scope='Local_value{}'.format(idx))

        self.worker_idx = idx

        self.env = env

        self.save_dir = save_dir

        self.Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        self.sess = sess
    def run(self):
        most_common_actions = []
        while Worker.global_episode < args.num_episodes:
            # Reset the environment and pick the first action
            self.env.reset()
            i_episode = 0
            episode = []
            Worker.episode_rewards.append(0)
            Worker.episode_lengths.append(0)

            episode_reward = 0.0

            classified_image = self.env.getClassifiedDroneImage()

            done = False
            reward = 0
            actions = []
            num_steps = 0

            for t in range(args.num_steps):

                # Take a step
                if t < args.seq_length:
                    action = np.random.randint(0, 4)
                else:
                    states = get_last_t_states(args.seq_length, episode)
                    action_probs = self.local_policy.predict(state=states, sess=self.sess)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                if t == 0:
                    last_action = action
                else:
                    last_action = actions[-1]

                next_image, reward, done = self.env.step(action, last_action)

                num_steps = t + 1

                actions.append(action)
                # Keep track of the transition
                episode.append(self.Transition(
                    state=classified_image, action=action, reward=reward, next_state=next_image, done=done))

                # Update statistics
                Worker.episode_rewards[Worker.global_episode] += reward
                Worker.episode_lengths[Worker.global_episode] = t
                episode_reward += reward

                # UPDATE EVERY TIMESTEP AFTER seq_length TIMESTEPS
                if t >= args.seq_length:
                    states = get_last_t_states(args.seq_length, episode)

                    states1 = get_last_t_minus_one_states(args.seq_length, episode)
                    value_next = self.local_value.predict(states1, sess=self.sess)

                    td_target = reward + args.gamma * value_next
                    td_error = td_target - self.local_value.predict(states, sess=self.sess)

                    #self.local_value.update(states, td_target)
                    #self.local_policy.update(states, td_error, action)

                    self.global_value.update(states, td_target, sess=self.sess)
                    self.global_policy.update(states, td_error, action, sess=self.sess)


                    update_policy_weights = [tf.assign(new, old) for (new, old) in
                                      zip(tf.trainable_variables('Local_policy{}'.format(self.worker_idx)), tf.trainable_variables('Global_policy'))]
                    update_value_weights = [tf.assign(new, old) for (new, old) in
                                      zip(tf.trainable_variables('Local_value{}'.format(self.worker_idx)), tf.trainable_variables('Global_value'))]

                    self.sess.run(update_policy_weights)
                    self.sess.run(update_value_weights)


                if done:
                    break

                classified_image = next_image


            Worker.global_moving_average_reward = record(Worker.global_episode, episode_reward,
                                                         self.worker_idx, Worker.global_moving_average_reward,
                                                         self.result_queue,
                                                         num_steps)

            Worker.global_episode += 1
            data = collections.Counter(actions)
            most_common_actions.append(data.most_common(1))
            print("Most common action: {}".format(most_common_actions[i_episode]))

            i_episode += 1
        self.result_queue.put(None)



parser = argparse.ArgumentParser(description='Drone A2C trainer')

parser.add_argument('--seq_length', default=6, type=int, help='number of stacked images')
parser.add_argument('--num_steps', default=1000, type=int, help='episode length')
parser.add_argument('--sight_dim', default=2, type=int, help='Drone sight distance')
parser.add_argument('--num_episodes', default=5000, type=int, help='number of episodes')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
parser.add_argument('--extra_layer', default=False, help='extra hidden layer between recurrent and output')
parser.add_argument('--num_workers', default=8, type=int, help='number of concurrent workers')
parser.add_argument('--save_dir', default='A3RCsaves', type=str, help='Save directory')
parser.add_argument('--layer_val', default=128, type=int, help='First layer hidden units (min=64')
parser.add_argument('--net_config', default=2, type=int, help='ID of network configuration')
parser.add_argument('--image', default='image1.JPG', type=str, help='Environment image file')
args = parser.parse_args()



env = Env(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

outF = open(os.path.join(args.save_dir, "hyper_parameters.txt"), "a")
write_hyper_parameters(outF)
outF.close()

with tf.Session() as sess:
    agent = MasterAgent(env)
    agent.train()
    

