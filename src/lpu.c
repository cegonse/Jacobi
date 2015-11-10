#include "jacobi.h"


double* getlpu(double *A, int n, double *LPU)
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
		__getlpu_kernel_sequential(A, n, LPU);
	}
	else
	{
		// Paralelo
		__getlpu_kernel_parallel(A, n, LPU, p);
	}
	
	return LPU;
}


static inline void __getlpu_kernel_sequential(double *A, int n, double *LPU)
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i != j) LPU[i + j*n] = A[i + j*n];
	    }
   	 }
}


static inline void __getlpu_kernel_parallel(double *A, int n, double *LPU, int p)
{
	int i, j;
	
	omp_set_num_threads(p);
	#pragma omp parallel for schedule(static) private(i, j) shared(A, n, LPU)
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i != j) LPU[i + j*n] = A[i + j*n];
	    }
   	 }
}

