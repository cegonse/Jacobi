#include "jacobi.h"


double* diaginv(double *A, int n, double *diag)
{
	int p, chunk, minsize;
	
	p = omp_get_max_threads();
	minsize = 4*p;
	chunk = n/p;
	
	#ifdef FORCE_SEQUENTIAL
	chunk = minsize - 1;
	#elif FORCE_OPENMP
	chunk = minsize + 1;
	#endif
	
	if (chunk < minsize)
	{
		// Secuencial
		__diaginv_kernel_sequential(A, n, diag);
	}
	else
	{
		// Paralelo
		__diaginv_kernel_parallel(A, n, diag, p);
	}
	
	return diag;
}


static inline void __diaginv_kernel_sequential(double *A, int n, double *diag)
{
	int i;
	
	for (i = 0; i < n; i++)
	{
		diag[i + n*i] =  1.0 / A[i + n*i];
	}
}


static inline void __diaginv_kernel_parallel(double *A, int n, double *diag, int p)
{
	int i;

	omp_set_num_threads(p);
	#pragma omp parallel for schedule(static) private(i) shared(A, n, diag)
	for (i = 0; i < n; i++)
	{
		diag[i + n*i] =  1.0 / A[i + n*i];
	}
}
