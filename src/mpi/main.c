#include <sys/time.h>
#include "../jacobi.h"

void saveMatrix(char *name, double *x, int n, int ndim)
{
	char fname[128];
	strcpy(fname, name);
	strcat(fname, ".dlm");

	FILE *f = fopen(fname, "w+");
	int i, j;
	
	if (f == NULL)
	{
		fprintf(stderr, "Error saving result: %s\n", strerror(errno));
		return;
	}
	
	if (ndim == 1)
	{
		for (i = 0; i < n; i++)
		{
			fprintf(f, "%.12f\n", x[i]);
		}
	}
	else if (ndim == 2)
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				fprintf(f, "%.12f ", x[i + j*n]);
			}
			
			fprintf(f, "\n");
		}
	}
	
	fclose(f);
}

int main (int argc, char* argv[])
{
	double conv = 0.01;
	int n, i, j, k, save = 0, rank, size, hlen;
	struct timeval t0, tf;
	double ep;
	char hname[MPI_MAX_PROCESSOR_NAME];
	double *A = NULL, *b = NULL, *x0 = NULL;
	
	MPI_Init (&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(hname, &hlen);
	
	if (argc < 2)
	{
		printf("Usage: jacobi [matrix_size] [--save]\n");
		
		MPI_Finalize();
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
			
			MPI_Finalize();
			exit(0);
		}
	}
	
	srand(time(NULL));
	n = atoi(argv[1]);
	
	// La inicialización del problema sólo se realiza en
	// el nodo 0
	if (rank == 0)
	{
		// Creamos las matrices del problema de tamaño 1*n o
		// n*n
		A = (double*) malloc(sizeof(double)*n*n);
		b = (double*) malloc(sizeof(double)*n);
		x0 = (double*) malloc(sizeof(double)*n);
		
		if (A == NULL || b == NULL || x0 == NULL)
		{
			fprintf(stderr, "Error creating input data: %s\n", strerror(errno));
			exit(1);
		}
		
		// Rellenamos A y b con valores aleatorios,
		// y x0 con todo unos
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				// Forzamos que A sea de diagonal dominante
				A[i + j*n] = 1.0 + (double)(rand() % 10);
				if (i == j) A[i + j*n] += (double)n*10.0;
			}
			
			b[i] = 1.0 + (double)(rand() % 10);
			x0[i] = 1.0;
		}
	}
	
	gettimeofday(&t0, NULL);
	k = jacobi_mpi(A, b, x0, conv, n, rank, size);
	gettimeofday(&tf, NULL);
	
	if (k < 0)
	{
		fprintf(stderr, "Error obtaining Jacobi solution on \"%s\": %s\n", hname, strerror(k));
	}
	
	ep = (tf.tv_sec - t0.tv_sec) + (tf.tv_usec - t0.tv_usec) / 1000000.0;
	
	// Borramos los datos inicializados al inicio del problema
	// y mostramos las estadísticas (tamaño matriz, iteraciones y tiempo).
	// Si se ha especificado, guardamos el resultado como archivo DLM
	if (rank == 0)
	{
		printf("%d %d %.6f\n", n, k, ep);
		
		if (save)
		{
			saveMatrix("x", x0, n, 1);
			saveMatrix("A", A, n, 2);
			saveMatrix("b", b, n, 1);
		}
		
		free(A);
		free(b);
		free(x0);
	}
	
	MPI_Finalize();
	
	return 0;
}
