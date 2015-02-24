
CC=gcc
CFLAGS+=--std=c99 -fopenmp

SIZE=10
THREADS=1

all: main utils test

main: main.c
	$(CC) $(CFLAGS) -o main main.c -lm

utils:
	$(CC) $(CFLAGS) -o datagen datagen.c
	$(CC) $(CFLAGS) -o serialtester serialtester.c -lm

test: main utils
	./datagen $(SIZE)
	./main $(SIZE) $(THREADS)
	./serialtester
