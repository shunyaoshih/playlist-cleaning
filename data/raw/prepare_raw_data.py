def read_file(file_name):
    input_file = open(file_name, 'r').read().splitlines()
    input_file = [seq.split(' ') for seq in input_file]
    return input_file

raw_file = read_file('./raw_data.txt')
rerank_file = read_file('./rerank_data.txt')

raw_dct = {}
rerank_dct = {}
for i, seq in enumerate(raw_file):
    key = seq[0] + ' ' + seq[1]
    raw_dct[key] = i
for i, seq in enumerate(rerank_file):
    key = seq[0] + ' ' + seq[1]
    rerank_dct[key] = i

x_file = open('./x.txt', 'w')
y_file = open('./y.txt', 'w')

for k, v in raw_dct.items():
    if k in rerank_dct:
        i = raw_dct[k]
        j = rerank_dct[k]
        x_file.write(' '.join(raw_file[i][2:]) + '\n')
        y_file.write(' '.join(rerank_file[j][2:]) + '\n')

x_file.close()
y_file.close()
