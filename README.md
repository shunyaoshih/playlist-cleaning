# playlist-cleaning

A cnn-based model for playlist cleaning and reranking integrated with the structure of super-resolution neural network.

This is one of my three projects during the internship at KKBOX. Note that
although there is a Python file related to seq2seq model, this project had not
supported the seq2seq model after an early experiment which shows that its
performance is much worse than the cnn-based model.

## Dependencies

* python3
* TensorFlow >= 1.2
* numpy
* tqdm

```
$ pip3 install -r requirements.txt
# to install TensorFlow, you can refer to https://www.tensorflow.org/install/
```

## Files you should prepare

### data/raw/x.txt

This file contains raw playlists.

```
[date_of_created_playlist] [song_id1] [song_id2] ....
```

### data/raw/y.txt

This file contains rereanked playlists corresponding to data/raw/x.txt line by line.

```
[date_of_created_playlist] [song_id1] [song_id2] ....
```

## Usage

### Prepare data
```
$ ./prepare_data.sh
```
### Train
```
$ python3 main.py --nn cnn --mode train
```

### Valid
```
$ python3 main.py --nn cnn --mode valid
```

### Create default testing set
```
$ ./create_default_test_data.sh
```
### Test
```
$ python3 main.py --nn cnn --mode test
```
