#include "jacobi.h"
#include "mathsub.h"


void getct(double *Dinv, double *LPU, double *T, double *b, double *C, int n)
{
	int p, minsize, chunk;
	
	p = omp_get_max_threads();
	minsize = 4*p;
	chunk = n/p;
	
	#ifdef FORCE_SEQUENTIAL
	chunk = minsize - 1;
	#elif FORCE_OPENMP
	chunk = minsize + 1;
	#endif
	
	// 1: T = -Dinv * LPU
	// 2: C = Dinv * b
	if (chunk < minsize)
	{
		// Secuencial
		__getct_kernel_sequential(Dinv, LPU, T, b, C, n);
	}
	else
	{
		// Paralelo
		__getct_kernel_parallel(Dinv, LPU, T, b, C, n, chunk, p);
	}
}


static inline void __getct_kernel_sequential(double *Dinv, double *LPU, double *T, double *b, double *C, int n)
{
	double alpha = -1.0, beta = 1.0, gamma = 0.0;
	
	dgemm_seq(n, alpha, Dinv, LPU, beta, T);
	dgemv_seq(n, beta, Dinv, b, gamma, C);
}

static inline void __getct_kernel_parallel(double *Dinv, double *LPU, double *T, double *b, double *C, int n, int chunk, int p)
{
	int incx = 1;
	char trans = 'N';
	double alpha = -1.0, beta = 1.0, gamma = 0.0;
	omp_set_num_threads(p);
	
	dgemm_(&trans, &trans, &n, &n, &n, &alpha, Dinv, &n, LPU, &n, &beta, T, &n);
	dgemv_(&trans, &n, &n, &beta, Dinv, &n, b, &incx, &gamma, C, &incx);
}