""" main function """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.utils import read_testing_sequences, word_id_to_song_id
from lib.utils import reward_functions
from lib.multi_task_seq2seq_model import Multi_Task_Seq2Seq
from lib.srcnn_model import SRCNN

def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config

if __name__ == "__main__":
    para = params_setup()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -para.init_weight, para.init_weight
        )
        if para.nn == 'rnn':
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                model = Multi_Task_Seq2Seq(para)
        elif para.nn == 'cnn':
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                model = SRCNN(para)

        try:
            os.makedirs(para.model_dir)
        except os.error:
            pass
        print(para)
        sv = tf.train.Supervisor(logdir=para.model_dir)
        with sv.managed_session(config=config_setup()) as sess:
            para_file = open('%s/para.txt' % (para.model_dir), 'w')
            para_file.write(str(para))
            para_file.close()
            if para.mode == 'train':
                step_time = 0.0
                for step in range(20000):
                    if sv.should_stop():
                        break
                    start_time = time.time()
                    [loss, predict_count, _] = sess.run([
                        model.loss,
                        model.predict_count,
                        model.update,
                    ])

                    loss = loss * para.batch_size
                    perplexity = np.exp(loss / predict_count)

                    step_time += (time.time() - start_time)
                    if step % para.steps_per_stats == 0:
                        print('step: %d, perplexity: %.2f step_time: %.2f' %
                            (step, perplexity, step_time / para.steps_per_stats))
                        step_time = 0
                    break

            elif para.mode == 'rl':
                step_time = 0.0
                for step in range(20000):
                    if sv.should_stop():
                        break
                    start_time = time.time()

                    # get input data
                    print('get input data')
                    data = sess.run([
                        model.raw_encoder_inputs,
                        model.raw_encoder_inputs_len,
                        model.raw_seed_song_inputs,
                    ])
                    data = [e.astype(np.int32) for e in data]

                    # get sampled ids
                    print('get sampled ids')
                    [sampled_ids] = sess.run(
                        fetches=[
                            model.sampled_ids,
                        ],
                        feed_dict={
                            model.encoder_inputs: data[0],
                            model.encoder_inputs_len: data[1],
                            model.seed_song_inputs: data[2]
                        }
                    )
                    print('sampled_ids\' shape: {}'.format(sampled_ids.shape))

                    # get reward
                    print('get reward')
                    rewards = reward_functions(para, sampled_ids)

                    # feed rewards and update the model
                    print('update the model')
                    _ = sess.run(
                        fetches=[
                            model.rl_update,
                        ],
                        feed_dict={
                            model.encoder_inputs: encoder_inputs,
                            model.encoder_inputs_len: encoder_inputs_len,
                            model.seed_song_inputs: seed_song_inputs,
                            model.sampled_ids_inputs: sampled_ids,
                            model.rewards: rewards
                        }
                    )

                    step_time += (time.time() - start_time)
                    if step % para.steps_per_stats == 0:
                        print('reward: %.2f' % (np.mean(rewards)))
                        step_time = 0
                    break

            elif para.mode =='valid':
                for i in range(10):
                    [loss, predict_count] = sess.run([
                        model.loss,
                        model.predict_count,
                    ])
                    loss = loss * para.batch_size
                    perplexity = np.exp(loss / predict_count)
                    print('perplexity: %.2f' % perplexity)

            elif para.mode == 'test':
                encoder_inputs, encoder_inputs_len, seed_song_inputs = \
                    read_testing_sequences(para)

                [predicted_ids, decoder_outputs] = sess.run(
                    fetches=[
                        model.decoder_predicted_ids,
                        model.decoder_outputs,
                    ],
                    feed_dict={
                        model.encoder_inputs: encoder_inputs,
                        model.encoder_inputs_len: encoder_inputs_len,
                        model.seed_song_inputs: seed_song_inputs
                    }
                )
                print(predicted_ids.shape)

                output_file = open('results/{}_out.txt'.format(para.nn), 'w')
                output_file.write(word_id_to_song_id(para, predicted_ids))
                output_file.close()
