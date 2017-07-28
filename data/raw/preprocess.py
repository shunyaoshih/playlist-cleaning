from collections import defaultdict

_START_VOCAB = ['_PAD', '_BOS', '_EOS', '_UNK']

raw_file = open('./raw_data.txt', 'r').read().splitlines()
raw_seqs = [seq.split(' ') for seq in raw_file]

vocab = {}
for seq in raw_seqs:
    for word in seq:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
print('vocab size: {}'.format(len(vocab_list)))

output_file = open('../vocab_{}'.format(len(vocab_list)), 'w')
output_file.write('\n'.join(vocab_list))
output_file.close()

dct = defaultdict(lambda : 3, [[word, i] for i, word in enumerate(vocab_list)])

def song_to_id(file_name):
    input_file = open(file_name, 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in input_file]
    seqs = [[str(dct[word]) for word in seq] for seq in seqs]

    output_file_name = '../ids_' + file_name
    output_file = open(output_file_name, 'w')
    output_file.write('\n'.join([' '.join(seq) for seq in seqs]))
    output_file.close()

song_to_id('raw_data.txt')
song_to_id('rerank_data.txt')

input_file = open('../ids_rerank_data.txt', 'r').read().splitlines()
input_file = [seq.split(' ') for seq in input_file]

# check how many songs are not in raw playlists
counter = 0
gg = 0
for i in range(len(input_file)):
    for j in range(len(input_file[i])):
        if input_file[i][j] == '3':
            print("{}-th row: {}-th".format(i, j))
            gg += 1
        counter += 1
print('dedup not found rate: {}%'.format(gg / counter * 100))

