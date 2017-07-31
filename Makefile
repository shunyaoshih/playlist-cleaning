.PHONY: all debug test clean

all:
	clear
	python3 main.py

test:
	clear
	python3 main.py --mode test

debug:
	clear
	python3 main.py --debug 1

debug_test:
	clear
	python3 main.py --mode test --debug 1

clean:
	rm -r rnn_models
