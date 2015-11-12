#include "jacobi.h"
#include "mathsub.h"


void getct(double *Dinv, double *LPU, double *T, double *b, double *C, int n, int m)
{
	int p, minsize, chunk;
	
	p = omp_get_max_threads();
	minsize = BLOCK_SIZE;
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
		__getct_kernel_sequential(Dinv, LPU, T, b, C, n, m);
	}
	else
	{
		// Paralelo
		__getct_kernel_parallel(Dinv, LPU, T, b, C, n, m, chunk, p);
	}
}


static inline void 
__getct_kernel_sequential(double *Dinv, double *LPU, double *T, double *b, double *C, int n, int m)
{
	double alpha = -1.0, beta = 1.0, gamma = 0.0;
	
	// Absolutamente vergonzoso. Hay que refactorizar.
	#ifndef MPI
	dgemm_seq(n, m, alpha, Dinv, LPU, beta, T);
	dgemv_seq(n, m, beta, Dinv, b, gamma, C);
	#else
	int i, k, j, rank, size, stride;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	stride = (rank-1)*(m/(size-1))/2;
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			T[i + j*n] = 0.0;
			
			for (k = 0; k < m; k++) 
			{
				T[i + j*n] += -Dinv[i + k*m + stride] * LPU[k + j*m + stride];
			}
		}
    }
	
	for (i = 0; i < n; i++)
	{
		C[i] = 0.0;
		
		for (j = 0; j < m; j++)
		{
			C[i] += Dinv[i + j*m + stride] * b[j];
		}
	}
	#endif
}

static inline void 
__getct_kernel_parallel(double *Dinv, double *LPU, double *T, double *b, double *C, int n, int m, int chunk, int p)
{
	int incx = 1;
	char trans = 'N';
	double alpha = -1.0, beta = 1.0, gamma = 0.0;
	omp_set_num_threads(p);
	
	#ifndef MPI
	dgemm_(&trans, &trans, &n, &m, &n, &alpha, Dinv, &n, LPU, &n, &beta, T, &n);
	dgemv_(&trans, &n, &m, &beta, Dinv, &n, b, &incx, &gamma, C, &incx);
	#else
	int rank, size, stride;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	stride = (rank-1)*(m/(size-1))/2;
	
	dgemm_(&trans, &trans, &n, &m, &m, &alpha, Dinv + stride, &m, LPU + stride, &m, &beta, T, &n);
	dgemv_(&trans, &n, &m, &beta, Dinv + stride, &m, b, &incx, &gamma, C, &incx);
	#endif
}