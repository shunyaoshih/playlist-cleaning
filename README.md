# playlist-cleaning

A cnn-based model for playlist cleaning

## Dependencies

* python3
* TensorFlow >= 1.2
* numpy
* tqdm

```
$ pip3 install -r requirements.txt
$ # to install TensorFlow, you can refer to https://www.tensorflow.org/install/
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
