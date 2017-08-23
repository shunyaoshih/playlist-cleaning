total_num = 70000

_START_VOCAB = ['_PAD', '_BOS', '_EOS', '_UNK']

raw_file = open('./x.txt', 'r').read().splitlines()
raw_seqs = [seq.split(' ') for seq in raw_file]
raw_seqs = [seq[1:] for seq in raw_seqs]
raw_seqs = [[word for word in seq if word != 'None'] for seq in raw_seqs]
rerank_file = open('./y.txt', 'r').read().splitlines()
rerank_seqs = [seq.split(' ') for seq in rerank_file]
rerank_seqs = [seq[1:] for seq in rerank_seqs]
rerank_seqs = [[word for word in seq if word != 'None'] for seq in rerank_seqs]
raw_seqs.extend(rerank_seqs)
vocab = {}
for seq in raw_seqs:
    for word in seq:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
# vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
vocab_list = sorted(vocab.items(), key=lambda t: t[::-1], reverse=True)
vocab_list = [vocab_list[i][0] for i in range(total_num - 4)]
vocab_list = _START_VOCAB + vocab_list
if len(vocab_list) > total_num:
    vocab_list = vocab_list[:total_num]
print('vocab size: {}'.format(len(vocab_list)))

output_file = open('../vocab_default.txt', 'w')
output_file.write('\n'.join(vocab_list))
output_file.close()

vocab = {}
for word in vocab_list:
    vocab[word] = 1

def read_file(file_name):
    input_file = open(file_name, 'r').read().splitlines()
    input_file = [seq.split(' ') for seq in input_file]
    input_file = [[seq[i] for i in range(len(seq))
                   if (seq[i] in vocab or i == 0 or i == 1) and seq[i] != 'None']
                  for seq in input_file if seq[1] in vocab]
    return input_file

raw_file = read_file('./x.txt')
rerank_file = read_file('./y.txt')

raw_dct = {}
rerank_dct = {}
for i, seq in enumerate(raw_file):
    key = seq[0] + ' ' + seq[1]
    raw_dct[key] = i
for i, seq in enumerate(rerank_file):
    key = seq[0] + ' ' + seq[1]
    rerank_dct[key] = i

x_file = open('./raw_data.txt', 'w')
y_file = open('./rerank_data.txt', 'w')

counter = 0
total_num = len(open('./x.txt', 'r').read().splitlines())
chosen_ids = []
for k, v in raw_dct.items():
    if k in rerank_dct and k in raw_dct:
        i = raw_dct[k]
        j = rerank_dct[k]
        if len(rerank_file[j]) < 32:
           continue
        chosen_ids.append(k)
        counter += 1
chosen_ids = sorted(chosen_ids, reverse=True)
for k in chosen_ids:
        i = raw_dct[k]
        j = rerank_dct[k]
        x_file.write(k + ' ')
        x_file.write(' '.join(raw_file[i][2:]) + '\n')
        y_file.write(k + ' ')
        y_file.write(' '.join(rerank_file[j][2:]) + '\n')
x_file.close()
y_file.close()
acc = 1.0 * counter / (1.0 * total_num) * 100.0
print('used data: {} / {} = {}%'.format(counter, total_num, counter / total_num * 100))
