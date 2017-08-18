""" convert input data to Standard Tensorflow Format """

import os
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict

max_len = 210

def create_meta_dct():
    input_file = open('./meta.txt', 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in input_file]
    dct = {}
    for seq in seqs:
        dct[int(seq[0])] = [int(seq[1]), int(seq[2])]
    dct[2] = [0, 0]
    return dct
def create_artist_and_genre_dct():
    def parse_file(file_name):
        input_file = open(file_name, 'r').read().splitlines()
        dct = defaultdict(lambda: 0)
        for i, ID in enumerate(input_file):
            dct[int(ID)] = i
        return dct
    return parse_file('./artist.txt'), parse_file('./genre.txt')

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _list_feature(lst):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=lst))

def convert_to_tf_format(mode):
    encoder_file = open('./{}_ids_raw_data.txt'.format(mode), 'r').read().splitlines()
    decoder_file = open('./{}_ids_rerank_data.txt'.format(mode), 'r').read().splitlines()
    seed_file = open('./{}_ids_seed.txt'.format(mode), 'r').read().splitlines()

    meta_dct = create_meta_dct()
    artist_dct, genre_dct = create_artist_and_genre_dct()
    vocab_list = open('./vocab_default.txt', 'r').read().splitlines()

    encoder_seqs = []
    encoder_seqs_len = []
    decoder_seqs = []
    decoder_seqs_len = []
    seed_ids = []

    artist_seqs = []
    genre_seqs = []
    seed_artist_seqs = []
    seed_genre_seqs = []

    for i in range(len(encoder_file)):
        encoder_seq_ids = encoder_file[i].strip().split(' ')
        decoder_seq_ids = decoder_file[i].strip().split(' ')

        encoder_seq_ids = [int(id) for id in encoder_seq_ids if len(id) > 0]
        decoder_seq_ids = [int(id) for id in decoder_seq_ids if len(id) > 0]

        artist_seqs.append(
            [artist_dct[meta_dct[int(vocab_list[id])][0]] for id in encoder_seq_ids]
        )
        genre_seqs.append(
            [genre_dct[meta_dct[int(vocab_list[id])][1]] for id in encoder_seq_ids]
        )
        seed_artist_seqs.append(
            artist_dct[meta_dct[int(vocab_list[int(seed_file[i])])][0]]
        )
        seed_genre_seqs.append(
            genre_dct[meta_dct[int(vocab_list[int(seed_file[i])])][1]]
        )

        encoder_seqs.append(encoder_seq_ids)
        encoder_seqs_len.append(len(encoder_seq_ids))
        decoder_seqs.append(decoder_seq_ids)
        # decoder_seqs_len.append(len(decoder_seq_ids))
        decoder_seqs_len.append(max_len)
        seed_ids.append(int(seed_file[i]))

    mx = max([max(encoder_seqs_len), max(decoder_seqs_len)])
    print('{}\'s max_len: {}'.format(mode, mx))
    mx = max_len
    encoder_seqs = [seq + [0] * (mx - len(seq)) for seq in encoder_seqs]
    decoder_seqs = [seq + [0] * (mx - len(seq)) for seq in decoder_seqs]
    artist_seqs = [seq + [0] * (mx - len(seq)) for seq in artist_seqs]
    genre_seqs = [seq + [0] * (mx - len(seq)) for seq in genre_seqs]
    print('num of data: %d' % (len(encoder_seqs)))
    print('max len: %d' % (len(decoder_seqs[0])))

    writer = tf.python_io.TFRecordWriter('cnn_{}.tfrecords'.format(mode))
    for i in tqdm(range(len(encoder_seqs))):
        example = tf.train.Example(features=tf.train.Features(feature={
            'encoder_input': _list_feature(encoder_seqs[i]),
            'encoder_input_len': _int64_feature(encoder_seqs_len[i]),
            'decoder_input': _list_feature(decoder_seqs[i]),
            'decoder_input_len': _int64_feature(decoder_seqs_len[i]),
            'seed_ids': _int64_feature(seed_ids[i]),
            'artist_input': _list_feature(artist_seqs[i]),
            'genre_input': _list_feature(genre_seqs[i]),
            'seed_artist_input': _int64_feature(seed_artist_seqs[i]),
            'seed_genre_input': _int64_feature(seed_genre_seqs[i]),
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    print('max_len should be less or equal to {}'.format(max_len))
    convert_to_tf_format('train')
    convert_to_tf_format('valid')
