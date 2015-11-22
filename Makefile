CC = icc
CFLAGS = -O3 -g -mkl -openmp -msse3
SRC = src/debug.c src/isdom.c src/rd.c src/mathsub.c src/tools.c src/standard/iter.c src/standard/jacobi.c src/standard/main.c
SEQ = -DFORCE_SEQUENTIAL
OMP = -DFORCE_OPENMP
DBG = -DDEBUG

all :
	$(CC) $(CFLAGS) $(SRC) -o bin/jacobi

force-sequential :
	$(CC) $(CFLAGS) $(SEQ) $(SRC) -o bin/jacobi
	
force-openmp :
	$(CC) $(CFLAGS) $(OMP) $(SRC) -o bin/jacobi
	
debug :
	$(CC) $(CFLAGS) $(DBG) $(SRC) -o bin/jacobi
	
debug-sequential :
	$(CC) $(CFLAGS) $(SEQ) $(DBG) $(SRC) -o bin/jacobi
	
debug-openmp :
	$(CC) $(CFLAGS) $(OMP) $(DBG) $(SRC) -o bin/jacobi
	
clean :
	rm -rf bin
	mkdir bin
