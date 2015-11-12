#include "../jacobi.h"
#include "../mathsub.h"


double jaciter(double *A, double *b, double *T, double *C, double *xk, double *xkp1, double *xconv, int n)
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
		e = __jaciter_kernel_sequential(A, b, T, C, xk, xkp1, xconv, n);
	}
	else
	{
		// Paralelo
		e = __jaciter_kernel_parallel(A, b, T, C, xk, xkp1, xconv, n, chunk, p);
	}
	
	return e;
}

double __jaciter_kernel_sequential(double *A, double *b, double *T, double *C, double *xk, double *xkp1, double *xconv, int n)
{
	double alpha = -1.0, beta = 1.0, gamma = 0.0, enode = 0.0;
	int rank, size, stride, m, i, j;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	m = n;
	n = m/(size-1);
	
	// x(k+1) = T*x(k) + C
	// 1: x(k+1) = T*x(k)
	for (i = 0; i < n; i++)
	{
		xkp1[i+(rank-1)*n] = 0.0;
		
		for (j = 0; j < m; j++)
		{
			xkp1[i+(rank-1)*n] += T[i + j*n] * xk[j];
		}
	}
	
	// 2: x(k+1) = x(k+1) + C
	for (i = 0; i < n; i++)
	{
		xkp1[i + n*(rank-1)] += C[i];
	}
	
	// Copiamos x(k+1) en x(k)
	dcopy_seq(n, xkp1+n*(rank-1), xk+n*(rank-1));
	
	return 0.0;
}


static inline double __jaciter_kernel_parallel(double *A, double *b, double *T, double *C, double *xk, double *xkp1,
double *xconv, int n, int chunk, int p)
{
	int incx = 1, rank, size, stride, m, i;
	char trans = 'N';
	double alpha = -1.0, beta = 1.0, gamma = 0.0, enode;
	
	omp_set_num_threads(p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	m = n;
	n = m/(size-1);
	stride = (rank-1)*(m/(size-1))/2;
	
	// x(k+1) = T*x(k) + C
	// 1: x(k+1) = T*x(k)
	dgemv_(&trans, &n, &m, &beta, T, &m, xk, &incx, &gamma, xkp1+n*(rank-1), &incx);
	// 2: x(k+1) = x(k+1) + C
	daxpy_(&n, &beta, C, &incx, xkp1+n*(rank-1), &incx);
	
	// Copiamos x(k+1) en x(k)
	//dcopy_(&n, xkp1, &incx, xk, &incx);
	dcopy_seq(m, xkp1, xk);
	
	// Calculamos la cota de error
	// e = norm2(A*x(k) - b)
	// 1: xconv = A*x(k)
	dgemv_(&trans, &n, &m, &beta, A, &n, xkp1, &incx, &gamma, xconv, &incx);
	
	// 2: xconv =  -b + xconv
	daxpy_(&n, &alpha, b+n*(rank-1), &incx, xconv, &incx);
	
	// Reducción
	for (i = 0; i < n; i++)
	{
		enode += abs(xconv[i])*abs(xconv[i]);
	}
	
	// 3: conv = norm2(xconv)
	return enode;
}
