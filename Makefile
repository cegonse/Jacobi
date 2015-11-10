CC = mpicc
CFLAGS = -O3 -g -mkl -openmp
SRC = src/debug.c src/diaginv.c src/isdom.c src/lpu.c src/ct.c src/mathsub.c src/iter.c src/jacobi.c src/main.c
SEQ = -DFORCE_SEQUENTIAL
OMP = -DFORCE_OPENMP
DBG = -DDEBUG

all : clean
	$(CC) $(CFLAGS) $(SRC) -o bin/jacobi

force-sequential : clean
	$(CC) $(CFLAGS) $(SEQ) $(SRC) -o bin/jacobi
	
force-openmp : clean
	$(CC) $(CFLAGS) $(OMP) $(SRC) -o bin/jacobi
	
debug : clean
	$(CC) $(CFLAGS) $(DBG) $(SRC) -o bin/jacobi
	
debug-sequential : clean
	$(CC) $(CFLAGS) $(SEQ) $(DBG) $(SRC) -o bin/jacobi
	
debug-openmp: clean
	$(CC) $(CFLAGS) $(OMP) $(DBG) $(SRC) -o bin/jacobi
	
clean :
	rm -rf bin
	mkdir bin
