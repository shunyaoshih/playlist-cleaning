""" a seq2seq model """

from copy import deepcopy

import numpy as np
import tensorflow as tf

import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense, dense
from tensorflow.python.util import nest

from lib.utils import read_num_of_seqs

__all__ = ['Sea2Seq']

class Seq2Seq():
    """ a seq2seq model """

    def __init__(self, para):
        self.para = para

        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        original_mode = deepcopy(self.para.mode)
        original_batch_size = deepcopy(self.para.batch_size)

        with tf.name_scope('train'):
            print('build training graph')
            self.para.mode = 'train'
            self.set_input()
            self.build_encoder()
            self.build_decoder()
            self.build_optimizer()

        tf.get_variable_scope().reuse_variables()
        self.para.batch_size = read_num_of_seqs()
        with tf.name_scope('test'):
            print('build testing graph')
            self.para.mode = 'test'
            self.set_input()
            self.build_encoder()
            self.build_decoder()

        self.para.mode = original_mode
        self.para.batch_size = original_batch_size

    def set_input(self):
        print('set input nodes...')
        if self.para.mode == 'train':
            self.raw_encoder_inputs, self.raw_encoder_inputs_len, \
            self.raw_decoder_inputs, self.raw_decoder_inputs_len = \
                self.read_batch_sequences()

            # self.encoder_inputs: [batch_size, max_len]
            self.encoder_inputs = self.raw_encoder_inputs[:, 1:]
            # self.encdoer_inputs_len: [batch_size]
            self.encoder_inputs_len = self.raw_encoder_inputs_len
            # self.decoder_inputs: [batch_size, decoder_max_len]
            self.decoder_inputs = self.raw_decoder_inputs[:, :-1]
            # self.decoder_inputs_len: [batch_size]
            self.decoder_inputs_len = self.raw_decoder_inputs_len
            # self.decoder_targets: [batch_size, max_len]
            self.decoder_targets = self.raw_decoder_inputs[:, 1:]

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

    def build_encoder(self):
        print('build encoder...')
        with tf.variable_scope('encoder'):
            self.encoder_cell = self.build_encoder_cell()

            self.encoder_embedding = tf.get_variable(
                name='embedding',
                shape=[self.para.encoder_vocab_size, self.para.embedding_size],
                dtype=self.dtype
            )
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embedding,
                ids=self.encoder_inputs
            )
            self.encoder_outputs, self.encoder_states = tf.nn.dynamic_rnn(
                cell=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_len,
                dtype=self.dtype,
            )

    def build_decoder(self):
        print('build decoder...')
        with tf.variable_scope('decoder'):
            self.decoder_cell, self.decoder_initial_state = \
                self.build_decoder_cell()

            self.decoder_embedding = tf.get_variable(
                name='embedding',
                shape=[self.para.decoder_vocab_size, self.para.embedding_size],
                dtype=self.dtype
            )
            output_projection_layer = Dense(
                units=self.para.decoder_vocab_size,
                name='output_projection'
            )

            if self.para.mode == 'train':
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embedding,
                    ids=self.encoder_inputs
                )

                if self.para.scheduled_sampling == 0:
                    training_helper = seq2seq.TrainingHelper(
                        inputs=self.decoder_inputs_embedded,
                        sequence_length=self.decoder_inputs_len,
                        name='training_helper'
                    )
                else:
                    self.sampling_probability = tf.cond(
                        self.global_step < self.para.start_decay_step * 2,
                        lambda: tf.cast(
                            tf.divide(self.global_step,
                                      self.para.start_decay_step * 2),
                            dtype=self.dtype),
                        lambda: tf.constant(1.0, dtype=self.dtype),
                        name='sampling_probability'
                    )
                    training_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                        inputs=self.decoder_inputs_embedded,
                        sequence_length=self.decoder_inputs_len,
                        embedding=self.decoder_embedding,
                        sampling_probability=self.sampling_probability,
                        name='training_helper'
                    )

                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state,
                    output_layer=output_projection_layer
                )
                max_decoder_length = tf.reduce_max(self.decoder_inputs_len)
                self.decoder_outputs, decoder_states, decoder_outputs_len = \
                    seq2seq.dynamic_decode(
                        decoder=training_decoder,
                        maximum_iterations=max_decoder_length
                    )

                rnn_output = self.decoder_outputs.rnn_output
                # rnn_output should be padded to max_len
                # calculation of loss will be handled by masks
                self.rnn_output_padded = tf.pad(rnn_output, \
                    [[0, 0],
                     [0, self.para.max_len - tf.shape(rnn_output)[1]],
                     [0, 0]] \
                )
                self.loss = self.compute_loss(
                    logits=self.rnn_output_padded,
                    labels=self.decoder_targets
                )

            elif self.para.mode == 'test':
                start_tokens = tf.fill([self.para.batch_size], 1)

                if self.para.beam_search == 0:
                    inference_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=start_tokens,
                        end_token=2,
                        embedding=self.decoder_embedding
                    )
                    inference_decoder = seq2seq.BasicDecoder(
                        cell=self.decoder_cell,
                        helper=inference_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=output_projection_layer
                    )
                else:
                    inference_decoder = seq2seq.BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=self.decoder_embedding,
                        start_tokens=start_tokens,
                        end_token=2,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.para.beam_width,
                        output_layer=output_projection_layer
                    )

                self.decoder_outputs, decoder_states, decoder_outputs_len = \
                    seq2seq.dynamic_decode(
                        decoder=inference_decoder,
                        maximum_iterations=self.para.max_len
                    )
                if self.para.beam_search == 0:
                    # self.decoder_predictions_id: [batch_size, max_len, 1]
                    self.decoder_predicted_ids = tf.expand_dims( \
                        input=self.decoder_outputs.sample_id, \
                        axis=-1 \
                    )
                else:
                    # self.decoder_predicted_ids: [batch_size, <= max_len, beam_width]
                    self.decoder_predicted_ids = self.decoder_outputs.predicted_ids


    def build_optimizer(self):
        print('build optimizer...')
        trainable_variables = tf.trainable_variables()
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
        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, \
                                                   self.para.max_gradient_norm)
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_variables),
            global_step=self.global_step
        )

    def compute_loss(self, logits, labels):
        """
            logits: [batch_size, max_len, decoder_vocab_size]
            labels: [batch_size, max_len, decoder_vocab_size]
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

    def sampled_softmax_loss(self):
        """ TODO: self-defined sampled softmax loss
        Args:
            self.rnn_output_padded: [batch_size, max_len, decoder_vocab_size]
            self.decoder_targets: [batch_size, max_len]
            self.masks: [batch_size, max_len]
            labels: [batch_size * max_len, num_samples(decoder_vocab_size)]
            (true prob distribution)
            logits: [batch_size * max_len, num_samples(decoder_vocab_size)]
        """
        loss = None

        return loss

    def build_encoder_cell(self):
        return tf.contrib.rnn.MultiRNNCell([self.build_single_cell()] * \
                                           self.para.num_layers)
    def build_decoder_cell(self):
        self.decoder_cell_list = \
           [self.build_single_cell() for i in range(self.para.num_layers)]

        if self.para.mode == 'train':
            encoder_outputs = self.encoder_outputs
            encoder_inputs_len = self.encoder_inputs_len
            encoder_states = self.encoder_states
            batch_size = self.para.batch_size
        else:
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs,
                multiplier=self.para.beam_width
            )
            encoder_inputs_len = seq2seq.tile_batch(
                self.encoder_inputs_len,
                multiplier=self.para.beam_width
            )
            encoder_states = seq2seq.tile_batch(
                self.encoder_states,
                multiplier=self.para.beam_width
            )
            batch_size = self.para.batch_size * self.para.beam_width

        if self.para.attention_mode == 'luong':
            # scaled luong: recommended by authors of NMT
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.para.num_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_len,
                scale=True
            )
            output_attention = True
        else:
            self.attention_mechanism = attention_wrapper.BahdanauAttention(
                num_units=self.para.num_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_len
            )
            output_attention = False

        cell = tf.contrib.rnn.MultiRNNCell(self.decoder_cell_list)
        cell = attention_wrapper.AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.para.num_units,
            name='attention'
        )
        decoder_initial_state = cell.zero_state(batch_size, self.dtype).clone(
            cell_state=encoder_states
        )

        return cell, decoder_initial_state

    def build_single_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.para.num_units)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=(1.0 - self.para.dropout)
        )
        return cell

    def read_batch_sequences(self):
        """ read a batch from .tfrecords """

        file_queue = tf.train.string_input_producer(['./data/train.tfrecords'])

        ei, ei_len, di, di_len = self.read_one_sequence(file_queue)

        min_after_dequeue = 3000
        capacity = min_after_dequeue + 3 * self.para.batch_size

        encoder_inputs, encoder_inputs_len, decoder_inputs, decoder_inputs_len = \
            tf.train.shuffle_batch(
                [ei, ei_len, di, di_len],
                batch_size=self.para.batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue
            )
        encoder_inputs = tf.sparse_tensor_to_dense(tf.to_int64(encoder_inputs))
        decoder_inputs = tf.sparse_tensor_to_dense(tf.to_int64(decoder_inputs))

        encoder_inputs_len = tf.reshape(encoder_inputs_len,
                                        [self.para.batch_size])
        decoder_inputs_len = tf.reshape(decoder_inputs_len,
                                        [self.para.batch_size])
        return encoder_inputs, tf.to_int32(encoder_inputs_len), \
               decoder_inputs, tf.to_int32(decoder_inputs_len)


    def read_one_sequence(self, file_queue):
        """ read one sequence from .tfrecords"""

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(file_queue)

        feature = tf.parse_single_example(serialized_example, features={
            'encoder_input': tf.VarLenFeature(tf.int64),
            'encoder_input_len': tf.FixedLenFeature([1], tf.int64),
            'decoder_input': tf.VarLenFeature(tf.int64),
            'decoder_input_len': tf.FixedLenFeature([1], tf.int64)
        })

        return feature['encoder_input'], feature['encoder_input_len'], \
               feature['decoder_input'], feature['decoder_input_len']
