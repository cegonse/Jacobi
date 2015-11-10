#include "../jacobi.h"
#include "../mathsub.h"


double jaciter(double *A, double *b, double *T, double *C, double *xk, double *xkp1, double *xconv, int n)
{
	int p, chunk, minsize;
	double e;
	
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
	int incx = 1;
	char trans = 'N';
	double alpha = -1.0, beta = 1.0, gamma = 0.0;

	// x(k+1) = T*x(k) + C
	// 1: x(k+1) = T*x(k)
	dgemv_seq(n, beta, T, xk, gamma, xkp1);
	
	// 2: x(k+1) = x(k+1) + C
	daxpy_seq(n, beta, C, xkp1);
	
	// Copiamos x(k+1) en x(k)
	dcopy_seq(n, xkp1, xk);
	
	// Calculamos la cota de error
	// e = norm2(A*x(k) - b)
	// 1: xconv = A*x(k)
	dgemv_seq(n, beta, A, xkp1, gamma, xconv);
	
	// 2: xconv =  -b + xconv
	daxpy_seq(n, alpha, b, xconv);
	
	// 3: conv = norm2(xconv)
	return dnrm2_seq(n, xconv);
}


static inline double __jaciter_kernel_parallel(double *A, double *b, double *T, double *C, double *xk, double *xkp1,
double *xconv, int n, int chunk, int p)
{
	int incx = 1;
	char trans = 'N';
	double alpha = -1.0, beta = 1.0, gamma = 0.0;
	
	// x(k+1) = T*x(k) + C
	// 1: x(k+1) = T*x(k)
	dgemv_(&trans, &n, &n, &beta, T, &n, xk, &incx, &gamma, xkp1, &incx);
	// 2: x(k+1) = x(k+1) + C
	daxpy_(&n, &beta, C, &incx, xkp1, &incx);
	
	// Copiamos x(k+1) en x(k)
	//dcopy_(&n, xkp1, &incx, xk, &incx);
	dcopy_seq(n, xkp1, xk);
	
	// Calculamos la cota de error
	// e = norm2(A*x(k) - b)
	// 1: xconv = A*x(k)
	dgemv_(&trans, &n, &n, &beta, A, &n, xkp1, &incx, &gamma, xconv, &incx);
	
	// 2: xconv =  -b + xconv
	daxpy_(&n, &alpha, b, &incx, xconv, &incx);
	
	// 3: conv = norm2(xconv)
	return dnrm2_(&n, xconv, &incx);
}
