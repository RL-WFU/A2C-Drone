import tensorflow as tf

class PolicyEstimator_RNN:
    def __init__(self, train_length, args, sess, policy_weight, sight_dim=2, batch_size = 1, lr=0.0001, scope='RNN_model_policy'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        self.args = args
        with tf.variable_scope(scope):

            self.state = tf.placeholder(shape=[None, self.train_length, (self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3],
                                        dtype=tf.float32, name='state')

            self.succ_visits = tf.placeholder(shape=[None, self.train_length, 4], dtype=tf.float32, name='Successor_visited')
            self.actions = tf.placeholder(shape=[None, self.train_length, 1], dtype=tf.float32, name='last_actions')


            if not self.args.feed_action:
                self.final_state = tf.concat([self.state, self.succ_visits], axis=2)
            else:
                self.state_and_pos = tf.concat([self.state, self.succ_visits], axis=2)
                self.final_state = tf.concat([self.state_and_pos, self.actions], axis=2)

            if self.args.net_config == 0:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=self.args.layer_val)
                self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=self.args.layer_val//2)

                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.args.layer_val//4)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2, initial_state=self.initial_state)

                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * self.args.layer_val//4])
                if self.args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=self.args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=4)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)

            elif self.args.net_config == 1:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=self.args.layer_val)
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.args.layer_val//2)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense1, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * self.args.layer_val//2])
                if self.args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=self.args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=4)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)

            elif self.args.net_config == 2:
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.args.layer_val)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * self.args.layer_val])
                if self.args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=self.args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=4)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)

            elif self.args.net_config == 3:
                self.x = tf.slice(self.final_state, [0, self.train_length - 1, 0], [-1, 1, -1])
                self.y = tf.squeeze(self.x, axis=1)
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.y, num_outputs=self.args.layer_val)
                self.output = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=4)





            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)
            self.picked_action_prob = tf.cond(self.picked_action_prob < 1e-30, lambda: tf.constant(1e-30), lambda: tf.identity(self.picked_action_prob))

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self.train_op = self.optimizer.minimize(self.loss)


            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)

            if self.args.test:
                self.saver.restore(sess, self.args.weight_dir + policy_weight)




    def predict(self, state, successor_visits, actions, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state, self.succ_visits : successor_visits, self.actions : actions})

    def update(self, states, target, action, successor_visits, actions, episode, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state: states, self.succ_visits : successor_visits, self.action: action, self.target: target, self.actions : actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

        if episode % (self.args.plot_interval * 4) == 0 and episode != 0 and not self.args.test:
            self.saver.save(sess, self.args.weight_dir + '/policy_weights', global_step=episode)

        return loss



class ValueEstimator_RNN:
    def __init__(self, train_length, args, sess, value_weight, sight_dim = 2, batch_size = 1, lr=0.0001, scope='RNN_model_value'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        self.args = args
        with tf.variable_scope(scope):

            self.state = tf.placeholder(shape=[None, self.train_length, (self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3],
                                        dtype=tf.float32, name='state')

            self.succ_visits = tf.placeholder(shape=[None, self.train_length, 4], dtype=tf.float32, name='Successor_visited')

            self.actions = tf.placeholder(shape=[None, self.train_length, 1], dtype=tf.float32, name='last_actions')

            if not self.args.feed_action:
                self.final_state = tf.concat([self.state, self.succ_visits], axis=2)
            else:
                self.state_and_pos = tf.concat([self.state, self.succ_visits], axis=2)
                self.final_state = tf.concat([self.state_and_pos, self.actions], axis=2)

            if self.args.net_config == 0:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=self.args.layer_val)
                self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=self.args.layer_val // 2)

                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.args.layer_val // 4)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2,
                                                            initial_state=self.initial_state)

                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * self.args.layer_val // 4])
                if self.args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=self.args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=1)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            elif self.args.net_config == 1:
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=self.args.layer_val)
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.args.layer_val // 2)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense1,
                                                            initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * self.args.layer_val // 2])
                if self.args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=self.args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=1)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            elif self.args.net_config == 2:
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.args.layer_val)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * self.args.layer_val])
                if self.args.extra_layer:
                    self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs,
                                                                    num_outputs=self.args.layer_val // 4)
                    self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=1)
                else:
                    self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            elif self.args.net_config == 3:
                self.x = tf.slice(self.final_state, [0, self.train_length - 1, 0], [-1, 1, -1])
                self.y = tf.squeeze(self.x, axis=1)
                self.dense1 = tf.contrib.layers.fully_connected(inputs=self.y, num_outputs=self.args.layer_val)
                self.output = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=1)



            #self.variables = tf.trainable_variables(scope)
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)
            if self.args.test:
                self.saver.restore(sess, self.args.weight_dir + value_weight)



            self.value_estimate = tf.squeeze(self.output)

            self.target = tf.placeholder(tf.float32, name='target')

            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, states, successor_visits, actions, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: states, self.succ_visits : successor_visits, self.actions : actions})

    def update(self, states, successor_visits, target, actions, episode, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: states, self.succ_visits : successor_visits, self.target: target, self.actions : actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        if episode % (self.args.plot_interval * 4) == 0 and episode != 0 and not self.args.test:
            self.saver.save(sess, self.args.weight_dir + '/value_weights', global_step=episode)

        return loss
