from collections import defaultdict

def create_vocabulary_file():
    vocab_list = open('../vocab_default.txt', 'r').read().splitlines()
    dct = defaultdict(lambda : 3, [[word, i] for i, word in enumerate(vocab_list)])
    return dct

def parse_train_or_valid_flag():
    raw_file = open('./raw_data.txt', 'r').read().splitlines()
    raw_file = [seq.split(' ') for seq in raw_file]
    raw_file = [seq[:2] for seq in raw_file]

    train_or_valid_flag = []
    dct = {}
    for i, seq in enumerate(raw_file):
        if seq[1] in dct:
            dct[seq[1]].append(seq[0])
            train_or_valid_flag.append(0)
        else:
            dct[seq[1]] = [seq[0]]
            train_or_valid_flag.append(1)
    return train_or_valid_flag

def song_id_to_vocab_id(file_name, train_or_valid_flag, vocab_dct):
    input_file = open(file_name, 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in input_file]
    seqs = [seq[2:] for seq in seqs]

    # song ids file
    output_train_file_name = '../train_' + file_name
    output_train_file = open(output_train_file_name, 'w')
    output_valid_file_name = '../valid_' + file_name
    output_valid_file = open(output_valid_file_name, 'w')
    output_test_file_name = '../test_' + file_name
    output_test_file = open(output_test_file_name, 'w')
    counter = 0
    for i, seq in enumerate(seqs):
        if counter < 32:
            output_test_file.write(' '.join(seq) + '\n')
            counter += 1
        elif train_or_valid_flag[i] == 1: # valid
            output_valid_file.write(' '.join(seq) + '\n')
        else: # train
            output_train_file.write(' '.join(seq) + '\n')
    output_train_file.close()
    output_valid_file.close()
    output_test_file.close()

    # song id to vocab id
    seqs = [[str(vocab_dct[word]) for word in seq] for seq in seqs]

    # vocab ids file
    output_train_file_name = '../train_ids_' + file_name
    output_train_file = open(output_train_file_name, 'w')
    output_valid_file_name = '../valid_ids_' + file_name
    output_valid_file = open(output_valid_file_name, 'w')
    output_test_file_name = '../test_ids_' + file_name
    output_test_file = open(output_test_file_name, 'w')
    counter = 0
    for i, seq in enumerate(seqs):
        if counter < 32:
            output_test_file.write(' '.join(seq) + '\n')
            counter += 1
        elif train_or_valid_flag[i] == 1: # valid
            output_valid_file.write(' '.join(seq) + '\n')
        else: # train
            output_train_file.write(' '.join(seq) + '\n')
    output_train_file.close()
    output_valid_file.close()
    output_test_file.close()

def create_vocab_id_seed_file(train_or_valid_flag, vocab_dct):
    raw_file = open('./raw_data.txt', 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in raw_file]
    seqs = [seq[1] for seq in seqs]

    # song ids file
    train_seed_file = open('../train_seed.txt', 'w')
    valid_seed_file = open('../valid_seed.txt', 'w')
    test_seed_file = open('../test_seed.txt', 'w')
    counter = 0
    for i, seq in enumerate(seqs):
        if counter < 32:
            test_seed_file.write(seq + '\n')
            counter += 1
        elif train_or_valid_flag[i] == 1:
            valid_seed_file.write(seq + '\n')
        else:
            train_seed_file.write(seq + '\n')
    train_seed_file.close()
    valid_seed_file.close()
    test_seed_file.close()

    # song id to vocab id
    seqs = [str(vocab_dct[word]) for word in seqs]

    # vocab ids file
    train_seed_file = open('../train_ids_seed.txt', 'w')
    valid_seed_file = open('../valid_ids_seed.txt', 'w')
    test_seed_file = open('../test_ids_seed.txt', 'w')
    counter = 0
    for i, seq in enumerate(seqs):
        if counter < 32:
            test_seed_file.write(seq + '\n')
            counter += 1
        elif train_or_valid_flag[i] == 1:
            valid_seed_file.write(seq + '\n')
        else:
            train_seed_file.write(seq + '\n')
    train_seed_file.close()
    valid_seed_file.close()
    test_seed_file.close()

def cal_dedup_id_not_found_rate():
    input_file = open('../train_ids_rerank_data.txt', 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in input_file]
    input_file = open('../valid_ids_rerank_data.txt', 'r').read().splitlines()
    input_file = [seq.split(' ') for seq in input_file]
    input_file.extend(seqs)

    # check how many songs are not in raw playlists
    counter = 0
    not_found = 0
    for i in range(len(input_file)):
        for j in range(len(input_file[i])):
            if input_file[i][j] == '3':
                # print("{}-th row: {}-th".format(i, j))
                not_found += 1
            counter += 1
    print('dedup not found rate: {}%'.format(not_found / counter * 100))

if __name__ == '__main__':
    vocab_dct = create_vocabulary_file()
    # 0 for training, 1 for validation
    train_or_valid_flag = parse_train_or_valid_flag()
    song_id_to_vocab_id('raw_data.txt', train_or_valid_flag, vocab_dct)
    song_id_to_vocab_id('rerank_data.txt', train_or_valid_flag, vocab_dct)

    create_vocab_id_seed_file(train_or_valid_flag, vocab_dct)

    cal_dedup_id_not_found_rate()
