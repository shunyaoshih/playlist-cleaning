""" main function """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.utils import read_testing_sequences, read_valid_sequences
from lib.utils import cal_precision_and_recall
from lib.utils import dict_id_to_song_id
from lib.utils import reward_functions
from lib.multi_task_seq2seq_model import Multi_Task_Seq2Seq
from lib.srcnn_model import SRCNN

prev_valid_loss = 10 ** 10
prev_precision = 0.0
prev_recall = 0.0

def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config

def load_weights(para, sess, model):
    # "xxxx_rl" => "rl"
    rl_mode = para.model_dir[len(para.model_dir) - 2:]
    if rl_mode != 'rl':
        ckpt = tf.train.get_checkpoint_state(para.model_dir)
        if ckpt:
            print('Loading model from %s' % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)

            # load prev_valid, prev_precision, prev_recall
            input_file = open(para.model_dir + '/result.txt', 'r').read().splitlines()
            input_file = [seq.split(' ') for seq in input_file]
            prev_valid_loss = float(input_file[0][1])
            prev_precision = float(input_file[1][1])
            prev_recall = float(input_file[2][1])
            print('prev_valid_loss: {}'.format(prev_valid_loss))
            print('prev_precision: {}'.format(prev_precision))
            print('prev_recall: {}'.format(prev_recall))
        else:
            print('Loading model with fresh parameters')
            sess.run(tf.global_variables_initializer())
    else:
        ckpt = tf.train.get_checkpoint_state(para.model_dir)
        if ckpt:
            print('Loading model from %s' % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # "xxxx_rl" => "xxxx"
            original_dir = para.model_dir[:len(para.model_dir) - 3]
            ckpt = tf.train.get_checkpoint_state(original_dir)
            if ckpt:
                print('Loading model from %s' % ckpt.model_checkpoint_path)
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Loading model with fresh parameters')
                sess.run(tf.global_variables_initializer())

def save_model(para, sess, model):
    [global_step] = sess.run([model.global_step])
    checkpoint_path = os.path.join(para.model_dir,
                                    "model.ckpt")
    model.saver.save(sess, checkpoint_path,
                        global_step=global_step)

if __name__ == "__main__":
    para = params_setup()

    if para.nn == 'rnn' and para.mode == 'rl':
        raise NameError('there is no support of RL on rnn')

    try:
        os.makedirs(para.model_dir)
    except os.error:
        pass
    para_file = open(para.model_dir + '/para.txt', 'w')
    para_file.write(str(para))
    para_file.close()

    print(para)

    graph = tf.Graph()
    with graph.as_default():
        initializer = tf.random_uniform_initializer(
          -para.init_weight, para.init_weight
        )
        with tf.variable_scope('model', initializer=initializer):
            if para.nn == 'rnn':
                model = Multi_Task_Seq2Seq(para)
            elif para.nn == 'cnn':
                model = SRCNN(para)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='model')
    for var in variables:
        print('\t{}\t{}'.format(var.name, var.get_shape()))

    with tf.Session(config=config_setup(), graph=graph) as sess:
        # need to initialize variables no matter what you want to do later
        sess.run(tf.global_variables_initializer())

        load_weights(para, sess, model)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            if para.mode == 'train':
                step_time = 0.0
                for step in range(20000):
                    start_time = time.time()

                    [loss, predict_count, _] = sess.run(
                        fetches=[
                            model.loss,
                            model.predict_count,
                            model.update,
                        ],
                    )

                    loss = loss * para.batch_size
                    perplexity = np.exp(loss / predict_count)

                    step_time += (time.time() - start_time)
                    if step % para.steps_per_stats == 0:
                        print('step: %d, perplexity: %.2f step_time: %.2f, ' %
                              (step, perplexity, step_time / para.steps_per_stats),
                              end='')

                        if para.nn == 'cnn':
                            encoder_inputs, seed_song_inputs, decoder_targets = \
                                read_valid_sequences(para)
                            [valid_loss, predicted_ids] = sess.run(
                                fetches=[
                                    model.valid_loss,
                                    model.valid_predicted_ids,
                                ],
                                feed_dict={
                                    model.valid_encoder_inputs: encoder_inputs,
                                    model.valid_seed_song_inputs: seed_song_inputs,
                                    model.valid_decoder_targets: decoder_targets,
                                }
                            )
                            print('valid perplexity: %.2f, ' % np.exp(valid_loss),
                                  end='')

                            precision, recall = cal_precision_and_recall(
                                predicted_ids, decoder_targets
                            )
                            print('precision: {}, recall, {} '.format(
                                precision, recall
                            ), end=' ')

                            # if precision + recall > prev_precision + prev_recall:
                            if valid_loss < prev_valid_loss:
                                prev_valid_loss = valid_loss
                                prev_precision = precision
                                prev_recall = recall
                                save_model(para, sess, model)
                                print(' --> save model to {}'.format(para.model_dir))

                                result_file = open(para.model_dir + '/result.txt', 'w')
                                result_file.write('perplexity: {}\n'.format(np.exp(prev_valid_loss)))
                                result_file.write('precision: {}\n'.format(precision))
                                result_file.write('recall: {}\n'.format(recall))
                                result_file.close()
                            else:
                                print()

                        step_time = 0
                    if para.debug:
                        break

            elif para.mode == 'rl':
                step_time = 0.0
                for step in range(20000):
                    start_time = time.time()

                    # get input data
                    data = sess.run([
                        model.raw_encoder_inputs,
                        model.raw_encoder_inputs_len,
                        model.raw_seed_song_inputs,
                    ])
                    data = [e.astype(np.int32) for e in data]

                    # get sampled ids
                    [sampled_ids] = sess.run(
                        fetches=[
                            model.sampled_ids,
                        ],
                        feed_dict={
                            model.encoder_inputs: data[0],
                            model.encoder_inputs_len: data[1],
                            model.seed_song_inputs: data[2],
                        }
                    )

                    # get reward
                    rewards, msg = reward_functions(para, sampled_ids)

                    # feed rewards and update the model
                    [_] = sess.run(
                        fetches=[
                            model.rl_update,
                        ],
                        feed_dict={
                            model.encoder_inputs: data[0],
                            model.encoder_inputs_len: data[1],
                            model.seed_song_inputs: data[2],
                            model.sampled_ids_inputs: sampled_ids,
                            model.rewards: rewards
                        }
                    )

                    step_time += (time.time() - start_time)
                    if step % para.steps_per_stats == 0:
                        print('sampled_ids: {}, reward: {}'.format(
                            sampled_ids[0][:30], rewards[0]) \
                        )
                        print('step: %d, rewards: %.2f step_time: %.2f' %
                              (step, np.mean(rewards), step_time / para.steps_per_stats),
                              end=' ')
                        print('length: %.2f' % (msg['length']), end=' ')
                        save_model(para, sess, model)
                        print(' --> save model to {}'.format(para.model_dir))
                        step_time = 0
                    if para.debug:
                        break

            elif para.mode =='valid':
                for i in range(5):
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
                        model.seed_song_inputs: seed_song_inputs,
                    }
                )

                output_file = open('results/{}_out.txt'.format(para.nn), 'w')
                output_file.write(dict_id_to_song_id(para, predicted_ids))
                output_file.close()

        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        finally:
            print('Stop')
            coord.request_stop()
            coord.join(threads)
