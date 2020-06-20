import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.layers import (TimeDistributed, Activation, Dense, Flatten, Input, GRU)
import numpy as np
from env_1 import *
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
    out.write("\n")
    out.write("feed_action: {}".format(args.feed_action))
    out.write("\n")
    out.write("plot_interval: {}".format(args.plot_interval))
    out.write("\n")
    out.write("env_dims: {}".format(args.env_dims))
    out.write("\n")
    out.write("binary_states: {}".format(args.binary_states))
    out.write("\n")
    out.write("binary_threshold: {}".format(args.binary_threshold))


class PolicyEstimator_RNN:
    def __init__(self, train_length, sight_dim=2, batch_size = 1, lr=0.0001, scope='RNN_model_policy'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        with tf.variable_scope(scope):

            self.state = tf.placeholder(shape=[None, self.train_length, (self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3],
                                        dtype=tf.float32, name='state')

            self.succ_visits = tf.placeholder(shape=[None, self.train_length, 4], dtype=tf.float32, name='Successor_visited')
            self.actions = tf.placeholder(shape=[None, self.train_length, 1], dtype=tf.float32, name='last_actions')


            if not args.feed_action:
                self.final_state = tf.concat([self.state, self.succ_visits], axis=2)
            else:
                self.state_and_pos = tf.concat([self.state, self.succ_visits], axis=2)
                self.final_state = tf.concat([self.state_and_pos, self.actions], axis=2)

            if args.net_config == 0:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=args.layer_val)
                self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=args.layer_val//2)

                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val//4)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2, initial_state=self.initial_state)

                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val//4])
                if args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=4)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)

            elif args.net_config == 1:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=args.layer_val)
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val//2)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense1, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val//2])
                if args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=4)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)

            elif args.net_config == 2:
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val])
                if args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=4)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)

            elif args.net_config == 3:
                self.x = tf.slice(self.final_state, [0, self.train_length - 1, 0], [-1, 1, -1])
                self.y = tf.squeeze(self.x, axis=1)
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.y, num_outputs=args.layer_val)
                self.output = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=4)





            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)


            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)

            if args.test:
                self.saver.restore(sess, args.weight_dir + policy_weight)




    def predict(self, state, successor_visits, actions, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state, self.succ_visits : successor_visits, self.actions : actions})

    def update(self, states, target, action, successor_visits, actions, episode, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state: states, self.succ_visits : successor_visits, self.action: action, self.target: target, self.actions : actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

        if episode % (args.plot_interval * 4) == 0 and episode != 0 and not args.test:
            self.saver.save(sess, args.weight_dir + '/policy_weights', global_step=episode)

        return loss

class ValueEstimator_RNN:
    def __init__(self, train_length, sight_dim = 2, batch_size = 1, lr=0.0001, scope='RNN_model_value'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        with tf.variable_scope(scope):

            self.state = tf.placeholder(shape=[None, self.train_length, (self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3],
                                        dtype=tf.float32, name='state')

            self.succ_visits = tf.placeholder(shape=[None, self.train_length, 4], dtype=tf.float32, name='Successor_visited')

            self.actions = tf.placeholder(shape=[None, self.train_length, 1], dtype=tf.float32, name='last_actions')

            if not args.feed_action:
                self.final_state = tf.concat([self.state, self.succ_visits], axis=2)
            else:
                self.state_and_pos = tf.concat([self.state, self.succ_visits], axis=2)
                self.final_state = tf.concat([self.state_and_pos, self.actions], axis=2)

            if args.net_config == 0:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=args.layer_val)
                self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=args.layer_val // 2)

                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val // 4)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2,
                                                            initial_state=self.initial_state)

                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val // 4])
                if args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=1)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            elif args.net_config == 1:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=args.layer_val)
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val // 2)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense1,
                                                            initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val // 2])
                if args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=1)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            elif args.net_config == 2:
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(args.layer_val)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * args.layer_val])
                if args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=1)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            elif args.net_config == 3:
                self.x = tf.slice(self.final_state, [0, self.train_length - 1, 0], [-1, 1, -1])
                self.y = tf.squeeze(self.x, axis=1)
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.y, num_outputs=args.layer_val)
                self.output = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=1)



            #self.variables = tf.trainable_variables(scope)
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)
            if args.test:
                self.saver.restore(sess, args.weight_dir + value_weight)



            self.value_estimate = tf.squeeze(self.output)

            self.target = tf.placeholder(tf.float32, name='target')

            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, states, successor_visits, actions, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: states, self.succ_visits : successor_visits, self.actions : actions})

    def update(self, states, successor_visits, target, actions, episode, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: states, self.succ_visits : successor_visits, self.target: target, self.actions : actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        if episode % (args.plot_interval * 4) == 0 and episode != 0 and not args.test:
            self.saver.save(sess, args.weight_dir + '/value_weights', global_step=episode)

        return loss


def get_last_t_states(t, episode):
    states = []
    actions = []
    succ_visits = []
    for i, transition in enumerate(episode[-t:]):
        states.append(transition.state)
        actions.append(transition.action)
        succ_visits.append(transition.succ_visit)


    succ_visits = np.asarray(succ_visits) #shape [t, 4]
    succ_visits = np.expand_dims(succ_visits, axis=0) #shape [1, t, 4]

    actions = np.asarray(actions)
    actions = np.expand_dims(actions, axis=0)
    actions = np.expand_dims(actions, axis=-1)

    states = np.asarray(states)
    states = np.expand_dims(states, axis=0)

    if args.binary_states:
        mining_binaries = np.zeros([1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        forest_binaries = np.zeros([1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        water_binaries = np.zeros([1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])

        mining = states[:, :, :, :, 0]
        mining = np.reshape(mining, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        forest = states[:, :, :, :, 1]
        forest = np.reshape(forest, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        water = states[:, :, :, :, 2]
        water = np.reshape(water, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])


        for i in range(t):
            for j in range((args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)):
                if mining[0, i, j] > args.binary_threshold:
                    mining_binaries[0, i, j] = 1
                if forest[0, i, j] > args.binary_threshold:
                    forest_binaries[0, i, j] = 1
                if water[0, i, j] > args.binary_threshold:
                    water_binaries[0, i, j] = 1

        binaries = np.concatenate([mining, forest, water], axis=2)

        return binaries, succ_visits, actions

    else:
        states = np.reshape(states, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1) * 3])
        return states, succ_visits, actions



def get_last_t_minus_one_states(t, episode):
    states = []
    actions = []
    succ_visits = []
    for i, transition in enumerate(episode[-t + 1:]):
        states.append(transition.state)
        actions.append(transition.action)
        succ_visits.append(transition.succ_visit)

    states.append(episode[-1].next_state)
    actions.append(episode[-1].action)
    succ_visits.append(episode[-1].succ_visit)

    succ_visits = np.asarray(succ_visits)
    succ_visits = np.expand_dims(succ_visits, axis=0)

    actions = np.asarray(actions)
    actions = np.expand_dims(actions, axis=0)
    actions = np.expand_dims(actions, axis=-1)

    states = np.asarray(states)
    states = np.expand_dims(states, axis=0)

    if args.binary_states:
        mining_binaries = np.zeros([1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        forest_binaries = np.zeros([1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        water_binaries = np.zeros([1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])

        mining = states[:, :, :, :, 0]
        mining = np.reshape(mining, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        forest = states[:, :, :, :, 1]
        forest = np.reshape(forest, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])
        water = states[:, :, :, :, 2]
        water = np.reshape(water, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)])

        for i in range(t):
            for j in range((args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)):
                if mining[0, i, j] > args.binary_threshold:
                    mining_binaries[0, i, j] = 1
                if forest[0, i, j] > args.binary_threshold:
                    forest_binaries[0, i, j] = 1
                if water[0, i, j] > args.binary_threshold:
                    water_binaries[0, i, j] = 1

        binaries = np.concatenate([mining, forest, water], axis=2)

        return binaries, succ_visits, actions

    else:
        states = np.reshape(states, [1, t, (args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1) * 3])
        return states, succ_visits, actions


def actor_critic(envs, estimator_policy, estimator_value, num_episodes, discount_factor=0.99):

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "succ_visit"])
    episode_rewards = np.zeros(0)
    episode_lengths = np.zeros(0)
    most_common_actions = []

    interval_averages = np.zeros(args.num_episodes // args.plot_interval)
    interval_totals = np.zeros(args.num_episodes // args.plot_interval)

    interval_avg_tot_covered = np.zeros(args.num_episodes // args.plot_interval)
    interval_avg_min_covered = np.zeros(args.num_episodes // args.plot_interval)
    interval_avg_wat_covered = np.zeros(args.num_episodes // args.plot_interval)
    interval_avg_for_covered = np.zeros(args.num_episodes // args.plot_interval)


    interval_avg_tot_mining = np.zeros(args.num_episodes // args.plot_interval)
    interval_avg_tot_mining_covered = np.zeros(args.num_episodes // args.plot_interval)

    total_mining = []
    total_mining_covered = []

    end_test = False

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action


        episode = []
        episode_rewards = np.append(episode_rewards, 0)
        episode_lengths = np.append(episode_lengths, 0)

        episode_env = envs[np.random.randint(0, len(envs))]

        episode_env.reset()

        classified_image = episode_env.getClassifiedDroneImage()


        done = False
        reward = 0
        actions = []

        successor_visit = np.zeros(4)

        for t in range(args.num_steps):

            # Take a step
            if t < args.seq_length:
                action = np.random.randint(0, 4)
            else:
                states, successor_visits, last_actions = get_last_t_states(args.seq_length, episode)
                action_probs = estimator_policy.predict(state=states, successor_visits=successor_visits, actions=last_actions)
                action_probs_mask = np.isnan(np.asarray(action_probs))
                valid=True
                for i in range(4):
                    if action_probs_mask[i]:
                        valid=False
                        break

                if valid:
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                else:
                    print(np.asarray(action_probs))
                    action = np.random.randint(0, 4)
                    end_test = True



            next_image, reward, done, successor_visit = episode_env.step(action)

            actions.append(action)
            # Keep track of the transition
            episode.append(Transition(
                state=classified_image, action=action, reward=reward, next_state=next_image, done=done, succ_visit = successor_visit))

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            #UPDATE EVERY TIMESTEP AFTER seq_length TIMESTEPS
            if t >= args.seq_length:
                states, successor_visits, last_actions = get_last_t_states(args.seq_length, episode)

                states1, successor_visits1, last_actions1 = get_last_t_minus_one_states(args.seq_length, episode)
                value_next = estimator_value.predict(states1, successor_visits1, last_actions1)

                td_target = reward + args.gamma * value_next
                td_error = td_target - estimator_value.predict(states, successor_visits, last_actions)

                estimator_value.update(states, successor_visits, td_target, last_actions, i_episode)
                estimator_policy.update(states, td_error, action, successor_visits, last_actions, i_episode)



            if done:
                break

            classified_image = next_image


        interval_totals[i_episode // args.plot_interval] += episode_rewards[i_episode]
        interval_averages[i_episode // args.plot_interval] = interval_totals[i_episode // args.plot_interval] / ((i_episode % args.plot_interval) + 1)

        percent_total = episode_env.totalCovered()
        percent_mining, tot_mining, tot_mining_covered = episode_env.miningCovered()
        percent_forest = episode_env.forestCovered()
        percent_water = episode_env.waterCovered()

        total_mining.append(tot_mining)
        total_mining_covered.append(tot_mining_covered)

        interval_avg_tot_covered[i_episode // args.plot_interval] = (interval_avg_tot_covered[i_episode // args.plot_interval] * (i_episode % args.plot_interval) + percent_total) / (i_episode % args.plot_interval + 1)
        interval_avg_min_covered[i_episode // args.plot_interval] = (interval_avg_min_covered[
                                                                             i_episode // args.plot_interval] * (
                                                                                     i_episode % args.plot_interval) + percent_mining) / (
                                                                                    i_episode % args.plot_interval + 1)
        interval_avg_for_covered[i_episode // args.plot_interval] = (interval_avg_for_covered[
                                                                             i_episode // args.plot_interval] * (
                                                                                     i_episode % args.plot_interval) + percent_forest) / (
                                                                                    i_episode % args.plot_interval + 1)
        interval_avg_wat_covered[i_episode // args.plot_interval] = (interval_avg_wat_covered[
                                                                             i_episode // args.plot_interval] * (
                                                                                     i_episode % args.plot_interval) + percent_water) / (
                                                                                    i_episode % args.plot_interval + 1)

        interval_avg_tot_mining[i_episode // args.plot_interval] = (interval_avg_tot_mining[
                                                                         i_episode // args.plot_interval] * (
                                                                                 i_episode % args.plot_interval) + tot_mining) / (
                                                                                i_episode % args.plot_interval + 1)
        interval_avg_tot_mining_covered[i_episode // args.plot_interval] = (interval_avg_tot_mining_covered[
                                                                         i_episode // args.plot_interval] * (
                                                                                 i_episode % args.plot_interval) + tot_mining_covered) / (
                                                                                i_episode % args.plot_interval + 1)

        if (i_episode + 1) % args.plot_interval == 0 and i_episode > 0:
            #episodes = [i + i_episode - args.plot_interval for i in range(args.plot_interval)]
            #plt.plot(episodes, episode_rewards[i_episode-args.plot_interval:i_episode])
            #plt.ylabel('Episode reward')
            #plt.xlabel('Episode')
            #plt.savefig(os.path.join(args.save_dir, 'Rewards {} to {}.png'.format(i_episode - args.plot_interval, i_episode)))
            #plt.clf()

            intervals = [i * args.plot_interval for i in range((i_episode//args.plot_interval) + 1)]
            plt.plot(intervals, interval_averages[0:(i_episode//args.plot_interval + 1)])
            plt.ylabel('Previous {} Episode Average'.format(args.plot_interval))
            plt.xlabel('Episode')
            plt.savefig(
                os.path.join(args.save_dir, 'Average Reward Episode {}'.format(i_episode)))
            plt.clf()

            episode_env.plot_visited(i_episode)

            plt.plot(intervals, interval_avg_tot_covered[0:(i_episode//args.plot_interval + 1)], color='black', label='Total')
            plt.plot(intervals, interval_avg_min_covered[0:(i_episode // args.plot_interval + 1)], color='red', linestyle='dashed',
                     label='Mining')
            plt.plot(intervals, interval_avg_for_covered[0:(i_episode // args.plot_interval + 1)], color='green', linestyle='dashed',
                     label='Forest')
            plt.plot(intervals, interval_avg_wat_covered[0:(i_episode // args.plot_interval + 1)], color='blue', linestyle='dashed',
                     label='Water')
            plt.xlabel('Episode')
            plt.ylabel('Previous {} Episode Average Coverage'.format(args.plot_interval))
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, 'Average Coverages Episode {}'.format(i_episode)))
            plt.clf()


            plt.plot(intervals, interval_avg_tot_mining[0:(i_episode//args.plot_interval + 1)], color='black', label='Total Mining')
            plt.plot(intervals, interval_avg_tot_mining_covered[0:(i_episode//args.plot_interval + 1)], color='black', label='Total Mining Covered', linestyle='dashed')
            plt.xlabel('Episode')
            plt.ylabel('Previous {} Episode Average Mining'.format(args.plot_interval))
            plt.savefig(os.path.join(args.save_dir, 'Mining Coverages Episode {}'.format(i_episode)))
            plt.legend()
            plt.clf()

            figs, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].hist(total_mining, 4)
            axs[1].hist(total_mining_covered, 4)
            plt.title("Left: Total Mining, Right: Mining Covered")
            plt.savefig(os.path.join(args.save_dir, 'Total Mining Histogram Episode {}'.format(i_episode)))
            plt.clf()




        data = collections.Counter(actions)
        most_common_actions.append(data.most_common(1))
        print("Episode {} finished. Reward: {}. Steps: {}. Most Common Action: {}.".format(i_episode,
                                                                                          episode_rewards[i_episode],
                                                                                          episode_lengths[i_episode],
                                                                                          most_common_actions[
                                                                                              i_episode]))


        if end_test:
            break



    plt.plot(episode_rewards)
    plt.ylabel('Episode reward')
    plt.xlabel('Episode')
    plt.savefig(os.path.join(args.save_dir, 'Rewards all episodes.png'))
    plt.clf()


def test(env, estimator_policy, estimator_value, num_episodes=200, sess=None):
    sess = sess or tf.get_default_session()

    Transition = collections.namedtuple("Transition",
                                        ["state", "action", "reward", "next_state", "done", "succ_visit"])

    episode_rewards = np.zeros(0)
    episode_lengths = np.zeros(0)
    total_coverage = np.zeros(0)
    total_mining = np.zeros(0)
    total_forest = np.zeros(0)
    total_water = np.zeros(0)
    most_common_actions = []

    for i_episode in range(num_episodes):
        row_pos, col_pos = env.reset()
        episode = []
        classified_image = env.getClassifiedDroneImage()
        actions = []

        episode_rewards = np.append(episode_rewards, 0)
        episode_lengths = np.append(episode_lengths, 0)


        for t in range(args.num_steps):

            if t < args.seq_length:
                action = np.random.randint(0, 4)
            else:
                states, succ_visits, last_actions = get_last_t_states(args.seq_length, episode)
                action_probs = estimator_policy.predict(state=states, successor_visits=succ_visits, actions=last_actions)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


            next_image, reward, done, succ_visit = env.step(action)

            actions.append(action)
            # Keep track of the transition
            episode.append(Transition(
                state=classified_image, action=action, reward=reward, next_state=next_image, done=done, succ_visit=succ_visit))

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t


            if done:
                break

            classified_image = next_image

        x, _, _ = env.miningCovered()

        total_coverage = np.append(total_coverage, env.totalCovered())
        total_mining = np.append(total_mining, x)
        total_forest = np.append(total_forest, env.forestCovered())
        total_water = np.append(total_water, env.waterCovered())

        data = collections.Counter(actions)
        most_common_actions.append(data.most_common(1))
        print("Episode {} finished. Reward: {}. Steps: {}. Most Common Action: {}".format(i_episode,

                                                                                          episode_rewards[i_episode],
                                                                                          episode_lengths[i_episode],
                                                                                          most_common_actions[
                                                                                              i_episode]))



    average_rewards = np.zeros(num_episodes // 10)
    average_total = np.zeros(num_episodes // 10)
    average_mining = np.zeros(num_episodes // 10)
    average_forest = np.zeros(num_episodes // 10)
    average_water = np.zeros(num_episodes // 10)

    for i in range(len(average_rewards)):
        average_rewards[i] = np.mean(episode_rewards[i * 10:i * 10 + 10])
        average_total[i] = np.mean(total_coverage[i * 10: i * 10 + 10])
        average_mining[i] = np.mean(total_mining[i * 10: i * 10 + 10])
        average_forest[i] = np.mean(total_forest[i * 10: i * 10 + 10])
        average_water[i] = np.mean(total_water[i * 10: i * 10 + 10])

    plt.plot(average_rewards)
    plt.ylabel('Average Reward (10 episodes)')
    plt.xlabel('Episode (in 10s)')
    plt.savefig(os.path.join(args.save_dir, 'Rewards TEST {} episodes.png'.format(num_episodes)))
    plt.clf()

    env.plot_visited(num_episodes)

    plt.plot(average_total, color='black', label='Total')
    plt.plot(average_mining, color='red', label='Mining', linestyle='dashed')
    plt.plot(average_forest, color='green', label='Forest', linestyle='dashed')
    plt.plot(average_water, color='blue', label='Water', linestyle='dashed')
    plt.ylabel('Average Coverages (Per 10 Episodes)')
    plt.xlabel('Episode (in 10s)')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'Coverages TEST {} episodes.png'.format(num_episodes)))
    plt.clf()




parser = argparse.ArgumentParser(description='Drone A2C trainer')

parser.add_argument('--seq_length', default=3, type=int, help='number of stacked images')
parser.add_argument('--num_steps', default=200, type=int, help='episode length')
parser.add_argument('--sight_dim', default=2, type=int, help='Drone sight distance')
parser.add_argument('--num_episodes', default=20000, type=int, help='number of episodes')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
parser.add_argument('--extra_layer', default=False, help='extra hidden layer between recurrent and output')
parser.add_argument('--save_dir', default='A2RC_saves/Test39', type=str, help='Save directory')
parser.add_argument('--layer_val', default=64, type=int, help='First layer hidden units (min=64')
parser.add_argument('--net_config', default=2, type=int, help='ID of network configuration')
parser.add_argument('--image', default='image4.jpg', type=str, help='Environment image file')
parser.add_argument('--feed_action', default=False, help='Whether to feed the last action or not')
parser.add_argument('--plot_interval', default=100, type=int, help='Number of episodes in between plotting')
parser.add_argument('--env_dims', default=25, type=int, help='Number of columns and rows in the environment')
parser.add_argument('--binary_states', default=False, help='Whether we input mining in binary format or not')
parser.add_argument('--binary_threshold', default=.5, type=float, help='Percentage at which a cell is considered mining')
parser.add_argument('--test', default=True, help='True for testing mode, false for training')
parser.add_argument('--weight_dir', default='A2RC_saves/Test39/Weights', type=str, help='Dir to save/get weights to/from')
args = parser.parse_args()





policy_weight = '/policy_weights-2800'
value_weight = '/value_weights-2800'


#environment = Env(args, args.image)

environments = []

if not args.test:
    img_dir = 'Env Images'
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".TIF"):
            environment = Env(args, os.path.join(img_dir, filename))
            environments.append(environment)
        else:
            continue

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.weight_dir):
    os.makedirs(args.weight_dir)

outF = open(os.path.join(args.save_dir, "hyper_parameters.txt"), "a")
write_hyper_parameters(outF)
outF.close()


with tf.Session() as sess:
    policy_estimator = PolicyEstimator_RNN(args.seq_length, args.sight_dim, lr=args.lr)
    value_estimator = ValueEstimator_RNN(args.seq_length, args.sight_dim, lr=args.lr)
    if not args.test:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        actor_critic(environments, policy_estimator, value_estimator, args.num_episodes)
    else:
        test_env = Env(args, args.image)
        test(test_env, policy_estimator, value_estimator)





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


#Give it binary info on whether it's been there or not


#Maps - go to google earth/maps