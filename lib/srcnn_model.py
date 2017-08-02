""" srcnn model """

from copy import deepcopy

import tensorflow as tf
from tensorflow.python.layers.core import dense

from lib.utils import read_num_of_lines

class SRCNN():
    def __init__(self, para):
        self.para = para

        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        original_mode = deepcopy(self.para.mode)
        original_batch_size = deepcopy(self.para.batch_size)

        self.build_weights()
        with tf.name_scope('train'):
            print('build training graph')
            self.para.mode = 'train'
            self.set_input()
            self.build_graph()
            self.build_optimizer()

        tf.get_variable_scope().reuse_variables()
        self.para.mode = 'test'
        self.para.batch_size = read_num_of_lines('results/in.txt')
        with tf.name_scope('test'):
            print('build testing graph')
            self.set_input()
            self.build_graph()

        self.para.mode = original_mode
        self.para.batch_size = original_batch_size

    def set_input(self):
        print('set input nodes...')
        if self.para.mode == 'train':
            self.raw_encoder_inputs, self.raw_encoder_inputs_len, \
            self.raw_decoder_inputs, self.raw_decoder_inputs_len, \
            self.raw_seed_song_inputs = self.read_batch_sequences()

            # self.encoder_inputs: [batch_size, max_len]
            self.encoder_inputs = self.raw_encoder_inputs
            # self.encdoer_inputs_len: [batch_size]
            self.encoder_inputs_len = self.raw_encoder_inputs_len
            # self.seed_song_inputs: [batch_size]
            self.seed_song_inputs = self.raw_seed_song_inputs
            # self.decoder_inputs: [batch_size, decoder_max_len]
            self.decoder_inputs = self.raw_decoder_inputs
            # self.decoder_inputs_len: [batch_size]
            self.decoder_inputs_len = self.raw_decoder_inputs_len
            # self.decoder_targets: [batch_size, max_len]
            self.decoder_targets = self.raw_decoder_inputs

            self.predict_count = tf.reduce_sum(self.decoder_inputs_len)
        elif self.para.mode == 'test':
            # self.encoder_inputs: [batch_size, max_len]
            self.encoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.para.max_len),
            )
            # encoder_inputs_length: [batch_size]
            self.encoder_inputs_len = tf.placeholder(
                dtype=tf.int32,
                shape=(None,)
            )
            # self.seed_song_inputs: [batch_size]
            self.seed_song_inputs = tf.placeholder(
                dtype=tf.int64,
                shape=(None,)
            )

    def build_graph(self):
        self.encoder_embedding = tf.get_variable(
            name='encoder_embedding',
            shape=[self.para.encoder_vocab_size, self.para.embedding_size],
            dtype=self.dtype
        )
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(
            params=self.encoder_embedding,
            ids=self.encoder_inputs
        )
        # self.encoder_inputs_embedded: [batch_size, max_len, embedding_size, 1]
        self.encoder_inputs_embedded = tf.reshape(
            self.encoder_inputs_embedded,
            [self.para.batch_size, self.para.max_len, self.para.embedding_size, 1]
        )
        inputs_shape = tf.shape(self.encoder_inputs_embedded)

        print(self.encoder_inputs_embedded.get_shape())
        conv1 = tf.nn.conv2d(
            input=self.encoder_inputs_embedded,
            filter=self.weights['w1'],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        if self.para.batch_norm == 1:
            conv1 = self.batch_normalization(
                conv1,
                self.offsets['o1'],
                self.scales['s1'],
                'conv1'
            )
        print(conv1.get_shape())
        conv1_relu = tf.nn.relu(conv1 + self.biases['b1'])
        conv1_relu = tf.nn.dropout(
            conv1_relu,
            keep_prob=(1.0 - self.para.dropout)
        )
        conv2 = tf.nn.conv2d(
            input=conv1_relu,
            filter=self.weights['w2'],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        if self.para.batch_norm == 1:
            conv2 = self.batch_normalization(
                conv2,
                self.offsets['o2'],
                self.scales['s2'],
                'conv2'
            )
        print(conv2.get_shape())
        conv2_relu = tf.nn.relu(conv2 + self.biases['b2'])
        conv2_relu = tf.nn.dropout(
            conv2_relu,
            keep_prob=(1.0 - self.para.dropout)
        )
        conv3 = tf.nn.conv2d(
            input=conv2_relu,
            filter=self.weights['w3'],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        if self.para.batch_norm == 1:
            conv3 = self.batch_normalization(
                conv3,
                self.offsets['o3'],
                self.scales['s3'],
                'conv3'
            )
        print(conv3.get_shape())
        conv3_relu = tf.nn.relu(conv3 + self.biases['b3'])
        conv3_relu = tf.nn.dropout(
            conv3_relu,
            keep_prob=(1.0 - self.para.dropout)
        )
        inv_conv3 = tf.nn.conv2d_transpose(
            conv3_relu,
            self.weights['inv_w3'],
            tf.shape(conv2_relu),
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        if self.para.batch_norm == 1:
            inv_conv3 = self.batch_normalization(
                inv_conv3,
                self.offsets['inv_o3'],
                self.scales['inv_s3'],
                'inv_conv3'
            )
        print(inv_conv3.get_shape())
        inv_conv3_relu = tf.nn.relu(inv_conv3 + self.biases['inv_b3'])
        inv_conv3_relu = tf.nn.dropout(
            inv_conv3_relu,
            keep_prob=(1.0 - self.para.dropout)
        )
        inv_conv2 = tf.nn.conv2d_transpose(
            inv_conv3_relu,
            self.weights['inv_w2'],
            tf.shape(conv1_relu),
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        if self.para.batch_norm == 1:
            inv_conv2 = self.batch_normalization(
                inv_conv2,
                self.offsets['inv_o2'],
                self.scales['inv_s2'],
                'inv_conv2'
            )
        print(inv_conv2.get_shape())
        inv_conv2_relu = tf.nn.relu(inv_conv2 + self.biases['inv_b2'])
        inv_conv2_relu = tf.nn.dropout(
            inv_conv2_relu,
            keep_prob=(1.0 - self.para.dropout)
        )
        inv_conv1 = tf.nn.conv2d_transpose(
            inv_conv2_relu,
            self.weights['inv_w1'],
            inputs_shape,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        if self.para.batch_norm == 1:
            inv_conv1 = self.batch_normalization(
                inv_conv1,
                self.offsets['inv_o1'],
                self.scales['inv_s1'],
                'inv_conv1'
            )
        print(inv_conv1.get_shape())
        inv_conv1_relu = tf.nn.relu(inv_conv1 + self.biases['inv_b1'])
        inv_conv1_relu = tf.nn.dropout(
            inv_conv1_relu,
            keep_prob=(1.0 - self.para.dropout)
        )
        self.residual_outputs = tf.add(
            inv_conv1_relu, self.encoder_inputs_embedded
        )
        self.embedding_outputs = tf.reshape(
            self.residual_outputs,
            [self.para.batch_size, self.para.max_len, self.para.embedding_size]
        )
        self.outputs = dense(
            inputs=self.embedding_outputs,
            units=self.para.decoder_vocab_size,
            name='output_projection'
        )
        if self.para.mode == 'train':
            self.loss = self.compute_loss(
                logits=self.outputs,
                labels=self.decoder_targets
            )
        else:
            # compatible with the rnn model
            self.decoder_outputs = self.outputs
            ids = tf.argmax(self.decoder_outputs, axis=2)
            self.decoder_predicted_ids = tf.reshape(
                ids,
                [self.para.batch_size, self.para.max_len, 1]
            )

    def build_optimizer(self):
        self.learning_rate = tf.cond(
           self.global_step < self.para.start_decay_step,
           lambda: tf.constant(self.para.learning_rate),
           lambda: tf.train.exponential_decay(
               self.para.learning_rate,
               (self.global_step - self.para.start_decay_step),
               self.para.decay_steps,
               self.para.decay_factor,
               staircase=True),
           name="learning_rate"
        )
        self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.update = self.opt.minimize(self.loss)

    def compute_loss(self, logits, labels):
        """
            logits: [batch_size, max_len, decoder_vocab_size]
            labels: [batch_size, max_len]
        """
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
        self.masks = tf.sequence_mask(
            lengths=self.decoder_inputs_len,
            maxlen=self.para.max_len,
            dtype=self.dtype,
            name='masks'
        )
        loss = tf.reduce_sum(crossent * self.masks) / \
               tf.to_float(self.para.batch_size)
        return loss

    def batch_normalization(self, input_tensor, offset, scale, name):
        """ global normalization """

        mean, variance = tf.nn.moments(input_tensor, [0, 1, 2])
        # print(mean.get_shape())
        # print(variance.get_shape())
        input_tensor_norm = tf.nn.batch_normalization(
            x=input_tensor,
            mean=mean,
            variance=variance,
            offset=offset,
            scale=scale,
            variance_epsilon=1e-8,
            name=name
        )
        return input_tensor_norm

    def read_batch_sequences(self):
        """ read a batch from .tfrecords """

        file_queue = tf.train.string_input_producer(['./data/cnn_train.tfrecords'])

        ei, ei_len, di, di_len, sid = self.read_one_sequence(file_queue)

        min_after_dequeue = 3000
        capacity = min_after_dequeue + 3 * self.para.batch_size

        encoder_inputs, encoder_inputs_len, decoder_inputs, decoder_inputs_len, \
        seed_ids = tf.train.shuffle_batch(
            [ei, ei_len, di, di_len, sid],
            batch_size=self.para.batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )
        encoder_inputs = tf.sparse_tensor_to_dense(encoder_inputs)
        decoder_inputs = tf.sparse_tensor_to_dense(decoder_inputs)

        encoder_inputs_len = tf.reshape(encoder_inputs_len,
                                        [self.para.batch_size])
        decoder_inputs_len = tf.reshape(decoder_inputs_len,
                                        [self.para.batch_size])
        return encoder_inputs, tf.to_int32(encoder_inputs_len), \
               decoder_inputs, tf.to_int32(decoder_inputs_len), seed_ids

    def read_one_sequence(self, file_queue):
        """ read one sequence from .tfrecords"""

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(file_queue)

        feature = tf.parse_single_example(serialized_example, features={
            'encoder_input': tf.VarLenFeature(tf.int64),
            'encoder_input_len': tf.FixedLenFeature([1], tf.int64),
            'decoder_input': tf.VarLenFeature(tf.int64),
            'decoder_input_len': tf.FixedLenFeature([1], tf.int64),
            'seed_ids': tf.FixedLenFeature([1], tf.int64)
        })

        return feature['encoder_input'], feature['encoder_input_len'], \
               feature['decoder_input'], feature['decoder_input_len'], \
               feature['seed_ids']

    def build_weights(self):
        self.weights = {
            'w1': tf.Variable(
                tf.random_normal([9, 9, 1, 64], stddev=1e-3),
                dtype=self.dtype,
                name='w1'
            ),
            'w2': tf.Variable(
                tf.random_normal([1, 1, 64, 32], stddev=1e-3),
                dtype=self.dtype,
                name='w1'
            ),
            'w3': tf.Variable(
                tf.random_normal([5, 5, 32, 1], stddev=1e-3),
                dtype=self.dtype,
                name='w1'
            ),
            'inv_w3': tf.Variable(
                tf.random_normal([5, 5, 32, 1], stddev=1e-3),
                dtype=self.dtype,
                name='inv_w3'
            ),
            'inv_w2': tf.Variable(
                tf.random_normal([1, 1, 64, 32], stddev=1e-3),
                dtype=self.dtype,
                name='inv_w2'
            ),
            'inv_w1': tf.Variable(
                tf.random_normal([9, 9, 1, 64], stddev=1e-3),
                dtype=self.dtype,
                name='inv_w1'
            ),
        }
        self.biases = {
            'b1': tf.Variable(
                tf.zeros([64]),
                dtype=self.dtype,
                name='b1'
            ),
            'b2': tf.Variable(
                tf.zeros([32]),
                dtype=self.dtype,
                name='b2'
            ),
            'b3': tf.Variable(
                tf.zeros([1]),
                dtype=self.dtype,
                name='b3'
            ),
            'inv_b3': tf.Variable(
                tf.zeros([32]),
                dtype=self.dtype,
                name='inv_b3'
            ),
            'inv_b2': tf.Variable(
                tf.zeros([64]),
                dtype=self.dtype,
                name='b2'
            ),
            'inv_b1': tf.Variable(
                tf.zeros([1]),
                dtype=self.dtype,
                name='inv_b1'
            ),
        }
        self.offsets = {
            'o1': tf.get_variable(
                name='o1',
                shape=[64],
                dtype=self.dtype
            ),
            'o2': tf.get_variable(
                name='o2',
                shape=[32],
                dtype=self.dtype
            ),
            'o3': tf.get_variable(
                name='o3',
                shape=[1],
                dtype=self.dtype
            ),
            'inv_o3': tf.get_variable(
                name='inv_o3',
                shape=[32],
                dtype=self.dtype
            ),
            'inv_o2': tf.get_variable(
                name='inv_o2',
                shape=[64],
                dtype=self.dtype
            ),
            'inv_o1': tf.get_variable(
                name='inv_o1',
                shape=[1],
                dtype=self.dtype
            ),
        }
        self.scales = {
            's1': tf.get_variable(
                name='s1',
                shape=[64],
                dtype=self.dtype
            ),
            's2': tf.get_variable(
                name='s2',
                shape=[32],
                dtype=self.dtype
            ),
            's3': tf.get_variable(
                name='s3',
                shape=[1],
                dtype=self.dtype
            ),
            'inv_s3': tf.get_variable(
                name='inv_s3',
                shape=[32],
                dtype=self.dtype
            ),
            'inv_s2': tf.get_variable(
                name='inv_s2',
                shape=[64],
                dtype=self.dtype
            ),
            'inv_s1': tf.get_variable(
                name='inv_s1',
                shape=[1],
                dtype=self.dtype
            ),
        }
