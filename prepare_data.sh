# !/bin/bash
cd data/raw
python3 prepare_raw_data.py
python3 prepare_ids_data.py
cd ..
python3 rnn_tf_format.py
python3 cnn_tf_format.py
cd ..
