#include "jacobi.h"

int jacobi(double *A, double *b, double *x0, double conv, int n)
{
	double *Dinv = NULL, *T = NULL, *C = NULL, *xkp1 = NULL,
	*xconv = NULL, *LPU = NULL, e = conv + 1.0, alpha = -1.0, beta = 1.0,
	gamma = 0.0;
	
	char trans = 'N';
	int i, j, k = 0, incx = 1;

	#ifdef DEBUG
	printf("A:\n");
	printMat(A, n);
	printf("\n\n");
	
	printf("b:\n");
	printVec(b, n);
	printf("\n\n");
	#endif
	
	// Ejecutar sólo si la matriz de términos independientes
	// es estrictamente dominante.
	if (!isdom(A, n)) return ENONDOM;
	
	// Reserva de memoria
	Dinv = (double*) malloc(sizeof(double)*n*n);
	T = (double*) malloc(sizeof(double)*n*n);
	LPU = (double*) malloc(sizeof(double)*n*n);
	C = (double*) malloc(sizeof(double)*n);
	xkp1 = (double*) malloc(sizeof(double)*n);
	xconv = (double*) malloc(sizeof(double)*n);
	
	if (Dinv == NULL || T == NULL || C == NULL || xkp1 == NULL || xconv == NULL || LPU == NULL)
	{
		return errno;
	}
	
	mkl_set_num_threads(1);
	
	// Obtenemos la inversa de la diagonal de A
	diaginv(A, n, Dinv);
	
	// Separamos la matriz A en L+U+D
	// Sólo almacenamos el resultado de L+U
	getlpu(A, n, LPU);
	
	#ifdef DEBUG
	printf("Dinv:\n");
	printMat(Dinv, n);
	printf("\n\n");
	
	printf("L+U:\n");
	printMat(LPU, n);
	printf("\n\n");
	#endif
	
	// Obtenemos las matrices T y C
	// Utilizando estas matrices, transformamos
	// la operación a la forma x1 = T*x0 + C
	//
	getct(Dinv, LPU, T, b, C, n);
	
	#ifdef DEBUG
	printf("T:\n");
	printMat(T, n);
	printf("\n\n");
	
	printf("C:\n");
	printVec(C, n);
	printf("\n\n");
	#endif // DEBUG
	
	mkl_set_num_threads(1);
	
	// Inicializamos x(k) y x(k+1) con el valor de x(0)
	dcopy_(&n, x0, &incx, xkp1, &incx);
	
	printf("Starting jacobi algorithm\n");
	
	// Iteramos hasta alcanzar la razón de convergencia
	while (e > conv)
	{
		// x(k+1) = T*x(k) + C
		// 1: x(k+1) = T*x(k)
		dgemv_(&trans, &n, &n, &beta, T, &n, x0, &incx, &gamma, xkp1, &incx);
		// 2: x(k+1) = x(k+1) + C
		daxpy_(&n, &beta, C, &incx, xkp1, &incx);
		
		// Copiamos x(k+1) en x(k)
		dcopy_(&n, xkp1, &incx, x0, &incx);
		
		// Calculamos la cota de error
		// e = norm2(A*x(k) - b)
		// 1: xconv = A*x(k)
		dgemv_(&trans, &n, &n, &beta, A, &n, x0, &incx, &gamma, xconv, &incx);
		
		// 2: xconv =  -b + xconv
		daxpy_(&n, &alpha, b, &incx, xconv, &incx);
		
		// 3: conv = norm2(xconv)
		e = dnrm2_(&n, xconv, &incx);
		
		#ifdef DEBUG
		printf("\n\nx(k+1) (k = %d):\n", k);
		printVec(xkp1, n);
		printf("\n\ne = %e / conv = %e\n", e, conv);
		#endif // DEBUG
		
		if (k % 10 == 0) printf("k = %d\n", k);
		
		k++;
	}
	
	// Liberamos la memoria utilizada
	free(Dinv);
	free(T);
	free(C);
	free(xkp1);
	
	return k;
}


