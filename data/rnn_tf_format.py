""" convert input data to Standard Tensorflow Format """

import os
import tensorflow as tf
from tqdm import tqdm

max_len = 210

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _list_feature(lst):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=lst))

def convert_to_tf_format(mode):
    encoder_file = open('./{}_ids_raw_data.txt'.format(mode), 'r').read().splitlines()
    decoder_file = open('./{}_ids_rerank_data.txt'.format(mode), 'r').read().splitlines()
    seed_file = open('./{}_ids_seed.txt'.format(mode), 'r').read().splitlines()

    encoder_seqs = []
    encoder_seqs_len = []
    decoder_seqs = []
    decoder_seqs_len = []
    seed_ids = []
    for i in range(len(encoder_file)):
        encoder_seq_ids = encoder_file[i].strip().split(' ')
        decoder_seq_ids = decoder_file[i].strip().split(' ')
        # if len(encoder_seq_ids) == 0 or len(encoder_seq_ids) > 199:
        #    continue
        # if len(decoder_seq_ids) == 0 or len(decoder_seq_ids) > 199:
        #    continue
        encoder_seq_ids = [1] + \
            [int(id) for id in encoder_seq_ids if len(id) > 0] + [2]
        decoder_seq_ids = [1] + \
            [int(id) for id in decoder_seq_ids if len(id) > 0] + [2]

        encoder_seqs.append(encoder_seq_ids)
        encoder_seqs_len.append(len(encoder_seq_ids) - 1)
        decoder_seqs.append(decoder_seq_ids)
        decoder_seqs_len.append(len(decoder_seq_ids) - 1)
        seed_ids.append(int(seed_file[i]))

    mx = max([max(encoder_seqs_len), max(decoder_seqs_len)]) + 1
    print('{}\'s max_len: {}'.format(mode, mx))
    mx = max_len
    encoder_seqs = [seq + [0] * (mx - len(seq)) for seq in encoder_seqs]
    decoder_seqs = [seq + [0] * (mx - len(seq)) for seq in decoder_seqs]
    print('num of data: %d' % (len(encoder_seqs)))
    print('max len: %d' % (len(decoder_seqs[0]) - 1))

    writer = tf.python_io.TFRecordWriter('rnn_{}.tfrecords'.format(mode))
    for i in tqdm(range(len(encoder_seqs))):
        example = tf.train.Example(features=tf.train.Features(feature={
            'encoder_input': _list_feature(encoder_seqs[i]),
            'encoder_input_len': _int64_feature(encoder_seqs_len[i]),
            'decoder_input': _list_feature(decoder_seqs[i]),
            'decoder_input_len': _int64_feature(decoder_seqs_len[i]),
            'seed_ids': _int64_feature(seed_ids[i])
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    # if not os.path.exists('./rnn_train.tfrecords'):
    print('max_len should be less or equal to {}'.format(max_len - 1))
    convert_to_tf_format('train')
    convert_to_tf_format('valid')
