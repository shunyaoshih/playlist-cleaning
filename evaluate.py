""" evaluation """

from lib.config import params_setup

def read_file(file_name):
    input_file = open(file_name, 'r').read().splitlines()
    input_file = [seq.split(' ') for seq in input_file]
    input_file = [[int(word) for word in seq] for seq in input_file]
    return input_file

def cal_precision(true_positives, false_positives):
    return (true_positives / (true_positives + false_positives))

def cal_recall(true_positives, false_negatives):
    return (true_positives / (true_positives + false_negatives))

def general_metrics(ans_txt, nn_txt, n):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(nn_txt)):
        limit = min([n, len(nn_txt[i])])
        for j in range(limit):
            if nn_txt[i][j] in ans_txt[i]:
                true_positives += 1
            else:
                false_positives += 1
    for i in range(len(ans_txt)):
        limit = min([n, len(ans_txt[i])])
        for j in range(limit):
            if ans_txt[i][j] not in nn_txt[i]:
                false_negatives += 1
    print('precision@{}: {}'.format(n, cal_precision(true_positives, false_positives)))
    print('recall@{}: {}'.format(n, cal_recall(true_positives, false_negatives)))

if __name__ == "__main__":
    para = params_setup()

    ans_path = './results/ans.txt'
    ans_txt = read_file(ans_path)

    nn_path = './results/' + para.nn + '_out.txt'
    nn_txt = read_file(nn_path)


    print('{}\'s result:'.format(para.nn))
    for i in range(10, 60, 10):
        general_metrics(ans_txt, nn_txt, i)
        print()
    general_metrics(ans_txt, nn_txt, 300)
