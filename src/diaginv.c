#include "jacobi.h"


double* diaginv(double *A, int n, double *diag)
{
	int p, chunk, csize;
	
	p = omp_get_max_threads();
	csize = 2*p;
	chunk = n/p;
	
	#ifdef FORCE_SEQUENTIAL
	csize = chunk - 1;
	#elif FORCE_OPENMP
	csize = chunk + 1;
	#endif
	
	if (chunk > csize)
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
