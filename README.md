# Jacobi

A parallel implementation of the Jacobi iteration to resolve
linear equation systems using OpenMP and MPI.

This application has been developed as a school project
and only has educational purposes.

First, a random linear equation system is created. In the standard
version, the data is equally distributed between the different threads
and the solution is obtained in parallel.

The MPI version distributes the data between the available compute nodes
and then obtains the solution in parallel.

Details on the implementation, algorithms, experimental results and
scalability are available on the PDF summary (in spanish). The raw
experimental results are available in the spreadsheet.

## Usage

Jacobi can be launched as a standalone application, using the
standard application or as a MPI process using the MPI application.

The input parameters for both applications are, in order:
* Matrix size. A random matrix of size NxN will be created to test
the algorithm.
* --save (optional). If present, the generated input data and the
result will be saved as .dlm files (dlm files can be read by Matlab
and Octave).

## Building

To build the standard version:

```
make
```

To build the MPI version:

```
make -f Makefile.mpi
```

For both versions, additional build targets can be used:
* debug: generate a debug build with additional error messages.
* force-sequential: force the application to use a single thread.
* force-parallel: force the application to use as many threads as
logical processors the machine has.

## Contributing

Jacobi is licensed under the GPL license. Feel free to contribute or to
use any part of this work on your own projects.