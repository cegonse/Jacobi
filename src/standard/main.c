#include "../jacobi.h"
#include <sys/time.h>
#include <locale.h>


int main (int argc, char* argv[])
{
	double conv = 0.001;
	int n, k = 0, save = 0;
	struct timeval t0, tf;
	double ep;
	double *A = NULL, *b = NULL, *x0 = NULL;
	
	if (argc < 2)
	{
		printf("Usage: jacobi [matrix_size] [--save]\n");
		exit(0);
	}
	else if (argc == 3)
	{
		if (strcmp("--save", argv[2]) == 0)
		{
			save = 1;
		}
		else
		{
			printf("Unrecognized option \"%s\".\n", argv[2]);
			printf("Usage: jacobi [matrix_size] [--save]\n");
			exit(0);
		}
	}
	
	n = atoi(argv[1]);
	
	A = (double*) malloc(sizeof(double)*n*n);
	b = (double*) malloc(sizeof(double)*n);
	x0 = (double*) malloc(sizeof(double)*n);
	
	// Creamos las matrices del problema
	k = generateMatrices(A, b, x0, n);
	
	if (k < 0)
	{
		fprintf(stderr, "Error generating input data: %s\n", strerror(k));
	}

	// Ejecutamos el problema
	gettimeofday(&t0, NULL);
	k = jacobi(A, b, x0, conv, n);
	gettimeofday(&tf, NULL);
	
	if (k < 0)
	{
		fprintf(stderr, "Error obtaining Jacobi solution: %s\n", strerror(k));
	}
	
	ep = (tf.tv_sec - t0.tv_sec) + (tf.tv_usec - t0.tv_usec) / 1000000.0;
	
	// Borramos los datos inicializados al inicio del problema
	// y mostramos las estadísticas (tamaño matriz, iteraciones y tiempo).
	// Si se ha especificado, guardamos el resultado como archivo DLM.
	setlocale(LC_NUMERIC, "es_ES.UTF-8");
	printf("%d;%d;%.6f\n", n, k, ep);
	
	if (save)
	{
		saveMatrix("x", x0, n, 1);
		saveMatrix("A", A, n, 2);
		saveMatrix("b", b, n, 1);
	}
	
	free(A);
	free(b);
	free(x0);
	
	return 0;
}
