#ifndef _JACOBI_H_
#define _JACOBI_H_

#include <math.h>
#include <malloc.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

enum
{
	ENONDOM = -1
};

int jacobi(double *A, double *b, double *x0, double conv, int n);

double* diaginv(double *A, int n, double *diag);
int isDominant(double *A, int n);

double* matmat(int isdiag, double *A, double *B, double *C, int n);
double* matvec(int isdiag, double *A, double *b, double *C, int n);
double* matscal(double* A, double s, int n);

void printMat(int isdiag, double* A, int n);
void printVec(double* a, int n);

#endif // _JACOBI_H_