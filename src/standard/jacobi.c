#include "../jacobi.h"
#include "../mathsub.h"


int jacobi(double *A, double *b, double *x0, double conv, int n)
{
	double *Dinv = NULL, *C = NULL, *xkp1 = NULL,
	*xconv = NULL, *R = NULL, e = conv + 1.0, alpha = -1.0, beta = 1.0,
	gamma = 0.0;
	
	int i, j, k = 0;

	// Ejecutar sólo si la matriz de términos independientes
	// es estrictamente dominante.
	if (!isdom(A, n))
	{
		return ENONDOM;
	}

	C = (double*) malloc(sizeof(double)*n);
	xkp1 = (double*) malloc(sizeof(double)*n);
	xconv = (double*) malloc(sizeof(double)*n);
	Dinv = (double*) malloc(sizeof(double)*n);
	R = (double*) malloc(sizeof(double)*n*n);
	
	// Ha ocurrido un error al reservar memoria
	if (Dinv == NULL || C == NULL || xkp1 == NULL || xconv == NULL || R == NULL)
	{
		return errno;
	}
	
	// Obtenemos R, Dinv y C
	getrd(Dinv, R, A, b, C, n, n);
	
	// Iteramos hasta alcanzar la razón de convergencia
	while (e > conv)
	{
		e = jaciter(A, b, R, C, Dinv, x0, xkp1, xconv, n);
		
		#ifdef DEBUG
		printf("(k = %d) e = %e / conv = %e\n", k, e, conv);
		#endif
			
		k++;
	}
	
	// Liberamos la memoria utilizada
	free(C);
	free(xkp1);
	free(xconv);
	free(Dinv);
	free(R);
	
	return k;
}


