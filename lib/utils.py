""" data processing functions """

import numpy as np
from copy import deepcopy
from collections import defaultdict
from math import sqrt
from random import sample

__all__ = ['dict_id_to_song_id',
           'read_valid_sequences',
           'read_testing_sequences',
           'read_num_of_lines',
           'get_max_len',
           'reward_functions',
           'cal_precision_and_recall']

dictionary_path = 'data/vocab_default.txt'

def read_dictionary():
    dict_file = open(dictionary_path, 'r').read().splitlines()
    dict_file = [(word, i) for i, word in enumerate(dict_file)]
    dic = defaultdict(lambda: 3)
    for word, idx in dict_file:
        dic[word] = idx
    return dic

def numpy_array_to_list(array):
    if isinstance(array, np.ndarray):
        return numpy_array_to_list(array.tolist())
    elif isinstance(array, list):
        return [numpy_array_to_list(element) for element in array]
    else:
        return array

def read_num_of_lines(file_name):
    seqs = open(file_name, 'r').read().splitlines()
    return len(seqs)

def get_max_len(file_name):
    input_file = open(file_name, 'r').read().splitlines()
    input_file = [seq.split(' ') for seq in input_file]
    return max([len(seq) for seq in input_file])

def read_valid_sequences(para):
    encoder_file = open('./data/valid_ids_raw_data.txt', 'r').read().splitlines()
    seed_file = open('./data/valid_ids_seed.txt', 'r').read().splitlines()
    target_file = open('./data/valid_ids_rerank_data.txt', 'r').read().splitlines()

    chosen_ids = sample(range(0, len(encoder_file)), para.batch_size)
    encoder_inputs = []
    seed_song_inputs = []
    decoder_targets = []
    for idx in chosen_ids:
        encoder_inputs.append(encoder_file[idx].split(' '))
        seed_song_inputs.append(seed_file[idx])
        decoder_targets.append(target_file[idx].split(' '))

    encoder_inputs = [seq + [0] * (para.max_len - len(seq)) for seq in \
                      encoder_inputs]
    seed_song_inputs = [int(num) for num in seed_song_inputs]
    decoder_targets = [seq + [0] * (para.max_len - len(seq)) for seq in \
                       decoder_targets]

    encoder_inputs = [[int(ID) for ID in seq] for seq in encoder_inputs]
    decoder_targets = [[int(ID) for ID in seq] for seq in decoder_targets]
    return np.asarray(encoder_inputs), np.asarray(seed_song_inputs), \
           np.asarray(decoder_targets)

def read_testing_sequences(para):
    # filter for smybol that utf8 cannot decode
    input_file = open('results/in.txt', 'r')
    output_file = open('results/in_filtered.txt', 'w')
    for line in input_file:
        output_file.write(bytes(line, 'utf-8').decode('utf-8', 'ignore'))
    input_file.close()
    output_file.close()
    seqs = open('results/in_filtered.txt', 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in seqs]

    for i in range(len(seqs)):
        if len(seqs[i]) > para.max_len - 1:
            seqs[i] = seqs[i][:para.max_len - 1]

    dic = read_dictionary()
    # input of seed ids
    seed_ids = open('results/seed.txt', 'r').read().splitlines()
    seed_ids = [dic[ID] for ID in seed_ids]

    seqs = [[dic[word] for word in seq] for seq in seqs]
    seqs = [seq + [2] for seq in seqs]

    seqs_len = [len(seq) for seq in seqs]
    seqs = [np.array(seq + [0] * (para.max_len - len(seq))) for seq in seqs]
    para.batch_size = len(seqs)
    print('total num of sequences: %d' % len(seqs))

    return np.asarray(seqs), np.asarray(seqs_len), np.asarray(seed_ids)

def check_valid_song_id(song_id):
    filter_list = [ 0, 1, 2, 3, -1]
    return not int(song_id) in filter_list

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not(x in seen or seen_add(x))]

def dict_id_to_song_id(para, predicted_ids):
    dic = open(dictionary_path, 'r').read().splitlines()
    # predicted_ids: [batch_size, <= max_len, beam_width]
    predicted_ids = numpy_array_to_list(predicted_ids)

    # song_id_seqs: [num_of_data * beam_width, <= max_len]
    song_id_seqs = []
    for seq in predicted_ids:
        for i in range(para.beam_width):
            song_id_seqs.append([seq[j][i] for j in range(len(seq))])

    # in cnn mode, we should discard all songs' ID which is after _EOS
    # tmp = []
    # if para.nn == 'cnn':
    #     now = []
    #     for seq in song_id_seqs:
    #         for song_id in seq:
    #             if song_id == 2:
    #                 break
    #             now.append(song_id)
    #         tmp.append(now)
    #         now = []
    # song_id_seqs = tmp

    song_id_seqs = [
       [dic[song_id] for song_id in seq if check_valid_song_id(song_id)]
       for seq in song_id_seqs
    ]

    song_id_seqs = [remove_duplicates(seq) for seq in song_id_seqs]

    return '\n'.join([' '.join(seq) for seq in song_id_seqs])

def cal_precision(true_positives, false_positives):
    return (true_positives / (true_positives + false_positives))

def cal_recall(true_positives, false_negatives):
    return (true_positives / (true_positives + false_negatives))

def cal_precision_and_recall(predicted_ids, targets):
    predicted_ids = numpy_array_to_list(predicted_ids)
    predicted_ids = [[ID[0] for ID in seq if check_valid_song_id(ID[0])] for seq in
                     predicted_ids]
    targets = [[ID for ID in seq if check_valid_song_id(ID)] for seq in
               targets]

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(predicted_ids)):
        now_set = set(targets[i])
        for j in range(len(predicted_ids[i])):
            if predicted_ids[i][j] in now_set:
                tp += 1
            else:
                fp += 1
    for i in range(len(targets)):
        now_set = set(predicted_ids[i])
        for j in range(len(targets[i])):
            if targets[i][j] not in now_set:
                fn += 1
    return cal_precision(tp, fp), cal_recall(tp, fn)

def length_reward(seq):
    return 1 - sqrt(abs(30 - len(seq)) / 150)

def reward_functions(para, sampled_ids):
    # approximate rewards' scale: -0.5 ~ 0.5
    rewards = [0.0] * para.batch_size
    msg = {}
    msg['length'] = 0

    sampled_ids = numpy_array_to_list(sampled_ids)
    song_id_seqs = sampled_ids
    song_id_seqs = [
       [song_id for song_id in seq if check_valid_song_id(song_id)]
       for seq in song_id_seqs
    ]

    for i in range(para.batch_size):
        rewards[i] = length_reward(song_id_seqs[i])
        msg['length'] += len(song_id_seqs[i])
    msg['length'] /= para.batch_size

    rewards = np.asarray(rewards, dtype=np.float32)
    rewards -= 0.5

    return rewards, msg
