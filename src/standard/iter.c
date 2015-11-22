#include "../jacobi.h"
#include "../mathsub.h"


double jaciter(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n)
{
	int p, chunk, minsize;
	double e;
	
	p = omp_get_max_threads();
	minsize = BLOCK_SIZE;
	chunk = n/p;
	
	#ifdef FORCE_SEQUENTIAL
	chunk = minsize - 1;
	#elif FORCE_OPENMP
	chunk = minsize + 1;
	#endif
	
	if (chunk < minsize)
	{
		// Secuencial
		e = __jaciter_kernel_sequential(A, b, R, C, Dinv, xk, xkp1, xconv, n);
	}
	else
	{
		// Paralelo
		e = __jaciter_kernel_parallel(A, b, R, C, Dinv, xk, xkp1, xconv, n, p);
	}
	
	return e;
}

static inline double 
__jaciter_kernel_sequential(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n)
{
	double e = 1.0;
	int i;
	
	dgemv_seq(n, n, 1.0, R, xk, 0.0, xkp1, 1);
		
	for (i = 0; i < n; i++)
	{
		xkp1[i] *= -Dinv[i];
		xkp1[i] += C[i];
	}
	
	dcopy_seq(n, xkp1, xk);
	
	// Norma
	dgemv_seq(n, n, 1.0, A, xk, 0.0, xconv, 1);
	daxpy_seq(n, -1.0, b, xconv, 1);
	dnrm2_seq(n, xconv, &e, 1);
	
	return e;
}


static inline double 
__jaciter_kernel_parallel(double *A, double *b, double *R, double *C, double *Dinv, double *xk, double *xkp1, double *xconv, int n, int p)
{
	double e = 1.0;
	int i;
	#ifdef FORCE_THREADS
	p = FORCE_THREADS;
	#endif
	omp_set_num_threads(p);
	
	dgemv_seq(n, n, 1.0, R, xk, 0.0, xkp1, p);
	
	#pragma omp parallel for schedule(static) private(i) shared(xkp1, Dinv, C, n)
	for (i = 0; i < n; i++)
	{
		xkp1[i] *= -Dinv[i];
		xkp1[i] += C[i];
	}
	
	dcopy_seq(n, xkp1, xk);
	
	// Norma
	dgemv_seq(n, n, 1.0, A, xk, 0.0, xconv, p);
	daxpy_seq(n, -1.0, b, xconv, p);
	dnrm2_seq(n, xconv, &e, p);
	
	return e;
}
