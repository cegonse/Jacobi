#include "../jacobi.h"
#include "../mathsub.h"

int jacobi_mpi(double *A, double *b, double *x0, double conv, int n, int rank, int size)
{
	double *Dinv = NULL, *T = NULL, *C = NULL, *xkp1 = NULL,
	*xconv = NULL, *LPU = NULL, e = conv + 1.0, alpha = -1.0, beta = 1.0,
	gamma = 0.0;
	
	int matdm = 0, k = 0, chunk = 0;
	MPI_Status status;
	
	#ifdef DEBUG
	if (rank == 0)
	{
		printf("A:\n");
		printMat(A, n);
		printf("\n\n");
	
		printf("b:\n");
		printVec(b, n);
		printf("\n\n");
	}
	#endif

	// Ejecutar sólo si la matriz de términos independientes
	// es estrictamente dominante.
	//
	// Sólo lo comprobamos en el nodo 0, en caso de no cumplir
	// la condición avisa al resto de nodos para que acaben todos
	if (rank == 0)
	{
		matdm = isdom(A, n);
		
		if (matdm)
		{	
			MPI_Abort(MPI_COMM_WORLD, ENONDOM);
		}
	}

	// Enviamos los trozos de la matriz A, del vector b
	// y del vector x0 que correspondan a cada nodo.
	//
	// Los nodos que no sean 0 recibirán los datos y
	// crearán las matrices a partir de los datos
	if (rank == 0)
	{
		// Calculamos el tamaño de bloque y lo
		// mandamos al resto de los nodos
		chunk = n / size;
		MPI_Bcast(&chunk, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else
	{
		// Esperamos a recibir el tamaño de bloque, y
		// reservamos la memoria necesaria para las
		// matrices auxiliares.
		MPI_Bcast(&chunk, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		C = (double*) malloc(sizeof(double)*chunk);
		xkp1 = (double*) malloc(sizeof(double)*chunk);
		xconv = (double*) malloc(sizeof(double)*chunk);
		Dinv = (double*) malloc(sizeof(double)*chunk*chunk);
		T = (double*) malloc(sizeof(double)*chunk*chunk);
		LPU = (double*) malloc(sizeof(double)*chunk*chunk);
		
		// Ha ocurrido un error al reservar memoria en uno de los nodos:
		// Cancelamos la ejecucción del programa.
		if (Dinv == NULL || T == NULL || C == NULL || xkp1 == NULL || xconv == NULL || LPU == NULL)
		{
			MPI_Abort(MPI_COMM_WORLD, errno);
		}
	}
	
	// Obtenemos la inversa de la diagonal de A
	Dinv = diaginv(A, n, Dinv);
	
	// Separamos la matriz A en L+U+D
	// Sólo almacenamos el resultado de L+U
	LPU = getlpu(A, n, LPU);
	
	#ifdef DEBUG
	if (rank == 0)
	{
		printf("Dinv:\n");
		printMat(Dinv, n);
		printf("\n\n");
		
		printf("L+U:\n");
		printMat(LPU, n);
		printf("\n\n");
	}
	#endif
	
	// Obtenemos las matrices T y C
	// Utilizando estas matrices, transformamos
	// la operación a la forma x1 = T*x0 + C
	//
	getct(Dinv, LPU, T, b, C, n);
	
	#ifdef DEBUG
	if (rank == 0)
	{
		printf("T:\n");
		printMat(T, n);
		printf("\n\n");
		
		printf("C:\n");
		printVec(C, n);
		printf("\n\n");
		
		printf("Starting Jacobi iteration.\n");
	}
	#endif // DEBUG
	
	// Inicializamos x(k+1) con el valor de x(0)
	dcopy_seq(n, x0, xkp1);
	
	// Iteramos hasta alcanzar la razón de convergencia
	while (e > conv)
	{
		e = jaciter(A, b, T, C, x0, xkp1, xconv, n);
		
		#ifdef DEBUG
		if (rank == 0)
		{
			printf("\n\nx(k+1) (k = %d):\n", k);
			printVec(xkp1, n);
			printf("\n\ne = %e / conv = %e\n", e, conv);
		}
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


