#include "jacobi.h"


void getct(double *Dinv, double *LPU, double *T, double *b, double *C, int n)
{
	int p, csize, chunk;
	
	p = omp_get_max_threads();
	csize = 2*p;
	chunk = n/p;
	
	#ifdef FORCE_SEQUENTIAL
	csize = chunk - 1;
	#elif FORCE_OPENMP
	csize = chunk + 1;
	#endif
	
	// 1: T = -Dinv * LPU
	// 2: C = Dinv * b
	if (csize > chunk)
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
	int incx = 1;
	char trans = 'N';
	double alpha = -1.0, beta = 1.0, gamma = 0.0;
	
	dgemm_seq(&n, &alpha, Dinv, LPU, &beta, T);
	dgemv_seq(&n, &beta, Dinv, b, &gamma, C);
}

static inline void __getct_kernel_parallel(double *Dinv, double *LPU, double *T, double *b, double *C, int n, int chunk, int p)
{
	int i, incx = 1;
	char trans = 'N';
	double alpha = -1.0, beta = 1.0, gamma = 0.0;
	
	omp_set_num_threads(p);
	
	#pragma omp parallel for schedule(static) private(i) shared(Dinv, LPU, T, trans, beta, alpha, incx, n, p, chunk)
	for (i = 0; i < p; i++)
	{
		dgemm_(&trans, &trans, &chunk, &n, &n, &alpha, Dinv + i*chunk, &n, LPU, &n, &beta, T + i*chunk, &n);
	}
	
	#pragma omp parallel for schedule(static) private(i) shared(Dinv, b, C, trans, beta, gamma, incx, n, p, chunk)
	for (i = 0; i < p; i++)
	{
		dgemv_(&trans, &chunk, &n, &beta, Dinv + i*chunk, &n, b + i*chunk, &incx, &gamma, C + i*chunk, &incx);
	}
}