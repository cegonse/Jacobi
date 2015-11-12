#include "../jacobi.h"
#include "../mathsub.h"

#include <stdio.h>

int jacobi(double *A, double *b, double *x0, double conv, int n)
{
	double *Dinv = NULL, *T = NULL, *C = NULL, *xkp1 = NULL,
	*xconv = NULL, *LPU = NULL, e = conv + 1.0, alpha = -1.0, beta = 1.0,
	gamma = 0.0;
	
	int k = 0;
	
	#ifdef DEBUG
	printf("A:\n");
	printMat(A, n, n);
	printf("\n\n");

	printf("b:\n");
	printVec(b, n);
	printf("\n\n");
	#endif

	// Ejecutar sólo si la matriz de términos independientes
	// es estrictamente dominante.
	if (!isdom(A, n))
	{
		return ENONDOM;
	}

	C = (double*) malloc(sizeof(double)*n);
	xkp1 = (double*) malloc(sizeof(double)*n);
	xconv = (double*) malloc(sizeof(double)*n);
	Dinv = (double*) malloc(sizeof(double)*n*n);
	T = (double*) malloc(sizeof(double)*n*n);
	LPU = (double*) malloc(sizeof(double)*n*n);
	
	// Ha ocurrido un error al reservar memoria
	if (Dinv == NULL || T == NULL || C == NULL || xkp1 == NULL || xconv == NULL || LPU == NULL)
	{
		return errno;
	}
	
	// Obtenemos la inversa de la diagonal de A
	Dinv = diaginv(A, n, Dinv);
	
	// Separamos la matriz A en L+U+D
	// Sólo almacenamos el resultado de L+U
	LPU = getlpu(A, n, LPU);
	
	#ifdef DEBUG
	printf("Dinv:\n");
	printMat(Dinv, n, n);
	printf("\n\n");
	
	printf("L+U:\n");
	printMat(LPU, n, n);
	printf("\n\n");
	#endif
	
	// Obtenemos las matrices T y C
	// Utilizando estas matrices, transformamos
	// la operación a la forma x1 = T*x0 + C
	//
	getct(Dinv, LPU, T, b, C, n, n);
	
	#ifdef DEBUG
	printf("T:\n");
	printMat(T, n, n);
	printf("\n\n");
	
	printf("C:\n");
	printVec(C, n);
	printf("\n\n");
	
	printf("Starting Jacobi iteration.\n");
	#endif // DEBUG
	
	// Inicializamos x(k+1) con el valor de x(0)
	dcopy_seq(n, x0, xkp1);
	
	// Iteramos hasta alcanzar la razón de convergencia
	while (e > conv)
	{
		e = jaciter(A, b, T, C, x0, xkp1, xconv, n);
		
		#ifdef DEBUG
		printf("\n\nx(k+1) (k = %d):\n", k);
		printVec(xkp1, n);
		printf("\n\ne = %e / conv = %e\n", e, conv);
		#endif // DEBUG
		
		k++;
	}
	
	// Liberamos la memoria utilizada
	free(C);
	free(xkp1);
	free(xconv);
	free(Dinv);
	free(T);
	free(LPU);
	
	return k;
}


