#ifndef _MATHSUB_H_
#define _MATHSUB_H_

void dgemm_seq(int n, double alpha, double *A, double *B, double beta, double *C);
double dnrm2_seq(int n, double *x);
void dgemv_seq(int n, double alpha, double *A, double *b, double beta, double *c);
void dcopy_seq(int n, double *x, double *y);
void daxpy_seq(int n, double alpha, double *x, double *y);


#endif // _MATHSUB_H_