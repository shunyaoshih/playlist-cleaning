""" main function """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.multi_task_seq2seq_model import Multi_Task_Seq2Seq

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
        else:
            # with tf.variable_scope('model', reuse=None, initializer=initializer):
            #     model = Seq2Seq(para)
            pass

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
                        model.update
                    ])

                    loss = loss * para.batch_size
                    perplexity = np.exp(loss / predict_count)

                    step_time += (time.time() - start_time)
                    if step % para.steps_per_stats == 0:
                        print('step: %d, perplexity: %.2f step_time: %.2f' %
                              (step, perplexity, step_time / para.steps_per_stats))
                        step_time = 0
                    step += 1

            elif para.mode == 'test':
                encoder_inputs, encoder_inputs_len = read_testing_sequences(para)

                [predicted_ids, decoder_outputs] = sess.run(
                    fetches=[
                        model.decoder_predicted_ids,
                        model.decoder_outputs,
                    ],
                    feed_dict={
                        model.encoder_inputs: encoder_inputs,
                        model.encoder_inputs_len: encoder_inputs_len
                    }
                )
                scores = cal_scores(
                    para,
                    predicted_ids,
                    decoder_outputs.beam_search_decoder_output.scores
                )
                print(predicted_ids.shape)
                print(scores)

                output_file = open('test/out.txt', 'w')
                output_file.write(word_id_to_song_id(para, predicted_ids))
                output_file.close()

                output_file = open('test/scores.txt', 'w')
                output_file.write('\n'.join(scores))
                output_file.close()
