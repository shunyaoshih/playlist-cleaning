""" arguments definition """

import argparse

def params_setup():
    """ arguments definition """

    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, default='rnn', help='')
    parser.add_argument('--mode', type=str, default='train', help='')

    # parameters for multi-task seq2seq
    parser.add_argument('--attention_mode', type=str, default='luong', help='')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='')
    parser.add_argument('--init_weight', type=float, default=0.1, help='')
    parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='')
    parser.add_argument('--num_units', type=int, default=128, help='')
    parser.add_argument('--num_layers', type=int, default=2, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--encoder_vocab_size', type=int, default=30000, help='')
    parser.add_argument('--decoder_vocab_size', type=int, default=86000, help='')
    parser.add_argument('--embedding_size', type=int, default=128, help='')
    parser.add_argument('--max_len', type=int, default=50, help='real_len + 1')
    parser.add_argument('--debug', type=int, default=0, help='')
    parser.add_argument('--beam_search', type=int, default=1, help='')
    parser.add_argument('--beam_width', type=int, default=0, help='')
    parser.add_argument('--num_samples', type=int, default=0, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument("--start_decay_step", type=int, default=0, help='')
    parser.add_argument('--decay_steps', type=int, default=10000, help='')
    parser.add_argument('--decay_factor', type=float, default=0.98, help='')
    parser.add_argument('--steps_per_stats', type=int, default=100, help='')
    parser.add_argument('--scheduled_sampling', type=int, default=1, help='')
    parser.add_argument('--model_dir', type=str, default='models', help='')

    # parameters for cnn

    para = parser.parse_args()

    para.encoder_vocab_size = len(open('data/vocab_default.in',
                                       'r').read().splitlines())
    para.decoder_vocab_size = len(open('data/vocab_default.ou',
                                       'r').read().splitlines())

    return para
