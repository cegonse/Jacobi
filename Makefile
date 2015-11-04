CC = gcc
LFLAGS = -Wall -lblas -lm
CFLAGS = -Wall -c -g -march=native -O3
OBJ = bin/objs/jacobi.o bin/objs/debug.o

all : clean jacobi.o debug.o main.o
	$(CC) $(LFLAGS) $(OBJ) bin/objs/main.o -o bin/jacobi

main.o :
	$(CC) $(CFLAGS) src/main.c -o bin/objs/main.o

jacobi.o :
	$(CC) $(CFLAGS) src/jacobi.c -o bin/objs/jacobi.o
	
debug.o :
	$(CC) $(CFLAGS) src/debug.c -o bin/objs/debug.o
	
clean :
	rm -rf bin
	mkdir bin
	mkdir bin/objs
