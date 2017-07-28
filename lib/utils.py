""" data processing functions """
import numpy as np
from copy import deepcopy
from collections import defaultdict

__all__ = ['word_id_to_song_id', 'read_testing_sequences', 'read_num_of_seqs']

encoder_vocab_path = 'data/vocab_default.in'
decoder_vocab_path = 'data/vocab_default.ou'

def str_to_bigram_list(seq):
    return [seq[i] + seq[i + 1] for i in range(len(seq) - 1)]

def read_dictionary(mode):
    if mode == 'encoder':
        file_path = encoder_vocab_path
    else:
        file_path = decoder_vocab_path
    dict_file = open(file_path, 'r').read().splitlines()
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

def read_num_of_seqs():
    seqs = open('test/in.txt', 'r').read().splitlines()
    return len(seqs)

def check_alternatives(word, alt_dict):
    if word in alt_dict:
        return alt_dict[word]
    return word

def read_testing_sequences(para):
    # filter for smybol that utf8 cannot decode
    input_file = open('test/in.txt', 'r')
    output_file = open('test/in_filtered.txt', 'w')
    for line in input_file:
        output_file.write(bytes(line, 'utf-8').decode('utf-8', 'ignore'))
    input_file.close()
    output_file.close()
    seqs = open('test/in_filtered.txt', 'r').read().splitlines()
    seqs = [str_to_bigram_list(seq) for seq in seqs]
    # for OOV
    alt_file = open('test/alternative_words.txt', 'r').read().splitlines()
    alt_list = [seq.split(' ') for seq in alt_file]
    alt_dict = defaultdict()
    for words in alt_list:
        alt_dict[words[0]] = words[1]
    seqs = [[check_alternatives(word, alt_dict) for word in seq] for seq in seqs]
    for seq in seqs:
        print(seq)

    dic = read_dictionary('encoder')
    seqs = [[dic[word] for word in seq] for seq in seqs]
    # filter for _UNK( unknown )
    seqs = [[ID for ID in seq if ID != 3] for seq in seqs]
    seqs = [seq + [2] for seq in seqs]
    if para.debug == 1:
        debug_dic = open(encoder_vocab_path, 'r').read().splitlines()
        for seq in seqs:
            seq = [debug_dic[word] for word in seq]
            print(seq)

    seqs_len = [len(seq) for seq in seqs]
    seqs = [np.array(seq + [0] * (para.max_len - len(seq))) for seq in seqs]
    para.batch_size = len(seqs)
    print('total num of sequences: %d' % len(seqs))

    return np.asarray(seqs), np.asarray(seqs_len)

def check_valid_song_id(song_id):
    filter_list = [ 0, 1, 2, 3, -1]
    return not song_id in filter_list

def word_id_to_song_id(para, predicted_ids):
    dic = open(decoder_vocab_path, 'r').read().splitlines()
    # predicted_ids: [batch_size, <= max_len, beam_width]
    predicted_ids = numpy_array_to_list(predicted_ids)

    # song_id_seqs: [num_of_data * beam_width, <= max_len]
    song_id_seqs = []
    for seq in predicted_ids:
        for i in range(para.beam_width):
            song_id_seqs.append([seq[j][i] for j in range(len(seq))])
    song_id_seqs = [
        [dic[song_id] for song_id in seq if check_valid_song_id(song_id)]
        for seq in song_id_seqs
    ]
    song_id_seqs = [list(set(seq)) for seq in song_id_seqs]

    # merge all beams
    tmp_seqs = deepcopy(song_id_seqs)
    song_id_seqs = []
    for i in range(int(len(tmp_seqs) / para.beam_width)):
       now = []
       for j in range(para.beam_width):
           now.extend(tmp_seqs[i * para.beam_width + j])
       song_id_seqs.append(list(set(now)))

    return '\n'.join([' '.join(seq) for seq in song_id_seqs])

def cal_scores(para, predicted_ids, scores):
    # predicted_ids: [batch_size, <= max_len, beam_width]
    predicted_ids = numpy_array_to_list(predicted_ids)
    # scores: [num_of_data, <= max_len, beam_width]
    scores = numpy_array_to_list(scores)

    for i in range(len(predicted_ids)):
        for j in range(len(predicted_ids[i])):
            for k in range(len(predicted_ids[i][j])):
                if check_valid_song_id(predicted_ids[i][j][k]) == False:
                    scores[i][j][k] = 0.0
    # beam_scores: [num_of_data * beam_width, max_len]
    beam_scores = []
    for seq in scores:
        for i in range(para.beam_width):
            beam_scores.append([seq[j][i] for j in range(len(seq))])
    num = [0] * len(beam_scores)
    for i in range(len(beam_scores)):
        for j in range(len(beam_scores[i])):
            if beam_scores[i][j] > 0.0:
                num[i] += 1
    beam_scores = [sum(seq) for seq in beam_scores]
    # normalized by the number of songs
    for i in range(len(beam_scores)):
        if num[i] > 0:
            beam_scores[i] = beam_scores[i] / num[i]

    final_scores = [0.0] * int(len(beam_scores) / para.beam_width)
    for i in range(int(len(beam_scores) / para.beam_width)):
        count = 0
        for j in range(para.beam_width):
            final_scores[i] += beam_scores[i * para.beam_width + j]
            if beam_scores[i * para.beam_width + j] != 0.0:
                count += 1
        if count > 0:
            final_scores[i] /= count

    final_scores = [str(score) for score in final_scores]

    return final_scores
