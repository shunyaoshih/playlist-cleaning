from collections import defaultdict

_START_VOCAB = ['_PAD', '_BOS', '_EOS', '_UNK']

vocab_list = open('../vocab_default.txt', 'r').read().splitlines()
dct = defaultdict(lambda : 3, [[word, i] for i, word in enumerate(vocab_list)])

def parse_train_and_valid():
    raw_file = open('./raw_data.txt', 'r').read().splitlines()
    raw_file = [seq.split(' ') for seq in raw_file]
    raw_file = [seq[:2] for seq in raw_file]

    train_or_valid = []
    dct = {}
    for i, seq in enumerate(raw_file):
        if seq[1] in dct:
            dct[seq[1]].append(seq[0])
            train_or_valid.append(0)
        else:
            dct[seq[1]] = [seq[0]]
            train_or_valid.append(1)
    return train_or_valid

def song_to_id(file_name, train_or_valid):
    input_file = open(file_name, 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in input_file]
    seqs = [seq[2:] for seq in seqs]
    # seqs = [[str(dct[word]) for word in seq] for seq in seqs]

    output_train_file_name = '../train_' + file_name
    output_train_file = open(output_train_file_name, 'w')
    output_valid_file_name = '../valid_' + file_name
    output_valid_file = open(output_valid_file_name, 'w')
    for i, seq in enumerate(seqs):
        if train_or_valid[i] == 1: # valid
            output_valid_file.write(' '.join(seq) + '\n')
        else: # train
            output_train_file.write(' '.join(seq) + '\n')
    output_train_file.close()
    output_valid_file.close()

train_or_valid = parse_train_and_valid() # 0 for training 1 for validation
song_to_id('raw_data.txt', train_or_valid)
song_to_id('rerank_data.txt', train_or_valid)

train_seed_file = open('../train_seed.txt', 'w')
valid_seed_file = open('../valid_seed.txt', 'w')
raw_file = open('./raw_data.txt', 'r').read().splitlines()
seqs = [seq.split(' ') for seq in raw_file]
seqs = [seq[1] for seq in seqs]
for i, seq in enumerate(seqs):
    if train_or_valid[i] == 1:
        valid_seed_file.write(seq + '\n')
    else:
        train_seed_file.write(seq + '\n')
train_seed_file.close()
valid_seed_file.close()
