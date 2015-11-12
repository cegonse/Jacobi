#include "../jacobi.h"
#include "../mathsub.h"

int jacobi_mpi(double *A, double *b, double *x0, double conv, int n, int rank, int size, char* hname)
{
	double *Dinv = NULL, *T = NULL, *C = NULL, *xkp1 = NULL, *xk = NULL,
	*xconv = NULL, *LPU = NULL, e = conv + 1.0, enode = 0.0, alpha = -1.0, beta = 1.0,
	gamma = 0.0;
	
	int matdm = 0, k = 0, chunk = 0, run = 1, stride, i = 0;
	MPI_Status status;

	#ifdef DEBUG
	if (rank == 0)
	{
		printf("A:\n");
		printMat(A, n, n);
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
	// la condición se cancela la ejecución
	if (rank == 0)
	{
		matdm = isdom(A, n);
		
		if (!matdm)
		{	
			fprintf(stderr, "Error obtaining Jacobi solution on \"%s\": 'A' matrix is non dominant.\n", hname);
			MPI_Abort(MPI_COMM_WORLD, ENONDOM);
		}
	}

	// Calculamos el tamaño de bloque. Dividimos el tamaño del 
	// problema (n) entre p - 1 nodos, ya que el nodo cero no
	// realiza procesamiento por bloques.
	chunk = n/(size-1);
	stride = (rank-1)*(n/(size-1))/2;
	
	#ifdef DEBUG
	if (rank == 0) printf("n: %d, size: %d, chunk: %d\n", n, size, chunk);
	#endif
	
	// Desde el nodo cero se envía la matriz A, los vectores
	// b y x0.
	if (rank == 0)
	{
		MPI_Bcast(A, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(x0, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		xconv = (double*) malloc(sizeof(double)*n);
	}
	else
	{
		// Esperamos a recibir el tamaño de bloque, y
		// reservamos la memoria necesaria para las
		// matrices auxiliares.
		MPI_Bcast(A, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(x0, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		C = (double*) malloc(sizeof(double)*chunk);
		T = (double*) malloc(sizeof(double)*chunk*n);
		xkp1 = (double*) malloc(sizeof(double)*n);
		xk = (double*) malloc(sizeof(double)*n);
		
		// Ha ocurrido un error al reservar memoria en uno de los nodos:
		// Cancelamos la ejecucción del programa.
		if (T == NULL || C == NULL || xkp1 == NULL || xk == NULL)
		{
			fprintf(stderr, "Error obtaining Jacobi solution on \"%s\": %s\n", hname, strerror(errno));
			MPI_Abort(MPI_COMM_WORLD, errno);
		}
	}
	
	// Las matrices D^-1 y L+U se obtienen en el nodo cero
	// y se envían a el resto de nodos.
	Dinv = (double*) malloc(sizeof(double)*n*n);
	LPU = (double*) malloc(sizeof(double)*n*n);
	
	// Ha ocurrido un error al reservar memoria en uno de los nodos:
	// Cancelamos la ejecucción del programa.
	if (Dinv == NULL || LPU == NULL)
	{
		fprintf(stderr, "Error obtaining Jacobi solution on \"%s\": %s\n", hname, strerror(errno));
		MPI_Abort(MPI_COMM_WORLD, errno);
	}
	
	// Obtenemos la inversa de la diagonal de A
	// y la matriz L+U en el nodo cero. Una vez
	// obtenidas, las mandamos al resto de nodos.
	if (rank == 0)
	{
		Dinv = diaginv(A, n, Dinv);
		LPU = getlpu(A, n, LPU);
		
		MPI_Bcast(Dinv, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(LPU, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Bcast(Dinv, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(LPU, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	
	// Obtenemos las matrices T y C
	// Utilizando estas matrices, transformamos
	// la operación a la forma x1 = T*x0 + C.
	//
	// En cada uno de los nodos se obtiene la parte
	// de la matriz T y del vector C que corresponda.
	if (rank != 0)
	{
		getct(Dinv, LPU, T, b, C, chunk, n);
		
		#ifdef DEBUG
		printf("C in %s (%d):\n", hname, rank);
		printVec(C, chunk);
		
		printf("\nT in %s (%d):\n", hname, rank);
		printMat(T, chunk, n);
		#endif
		
		// Inicializamos x(k+1) y x(k) con el valor de x(0).
		dcopy_seq(n, x0, xkp1);
		dcopy_seq(n, x0, xk);
	}
	
	// Iteramos hasta alcanzar la razón de convergencia.
	// Cada nodo obtendrá la parte que le corresponda de
	// la solución.
	//
	// Para obtener la razón de convergencia:
	// e = sqrt(sum(Ax-b)^2)
	//
	// Se calculará en cada nodo la parte correspondiente
	// de la operación de álgebra matricial, el cuadrado de
	// cada elemento y su suma.
	//
	// Posteriormente se enviará cada resultado al nodo cero,
	// el cual sumará todos los valores y obtendrá la raíz,
	// calculando así la norma del vector.
	//
	// Por tanto, el nodo cero será el encargado de decidir si se
	// debe continuar con la iteración o si se ha alcanzado la razón
	// de convergencia deseada. El resto de nodos esperarán su respuesta.
	while (run)
	{
		if (rank == 0)
		{
			// Recibimos el parámetro de convergencia de cada nodo y calculamos
			// la raíz cuadrada para obtener la 2-norma
			for (i = 0; i < size-1; i++)
			{
				MPI_Bcast(x0 + chunk*i, chunk, MPI_DOUBLE, i+1, MPI_COMM_WORLD);
			}
			
			char trans = 'N';
			double alpha = -1.0, beta = 1.0, gamma = 0.0, enode = 0.0;
			int incx = 1;
			
			dgemv_seq(n, n, beta, A, x0, gamma, xconv);
			daxpy_seq(n, alpha, b, xconv);
			dnrm2_seq(n, xconv, &e);
			
			#ifdef DEBUG
			printf("(k = %d) e = %e / conv = %e\n", k, e, conv);
			printVec(x0, n);
			#endif
			
			// Si aún no hemos alcanzado el valor de convergencia requerido,
			// continuamos.
			//
			// Avisamos al resto de nodos de si deben continuar o si deben
			// de acabar.
			if (e < conv) {
				run = 0;
			}
			else {
				k++;
			}
			
			MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		else
		{
			// Calculamos la solución local y mandamos el parámetro de convergencia
			jaciter(A + stride, b + stride, T, C, xk, xkp1, xconv, n);
			
			// Mandamos los bloques de x(k) al resto de nodos
			for (i = 0; i < size-1; i++)
			{
				MPI_Bcast(xk + chunk*i, chunk, MPI_DOUBLE, i+1, MPI_COMM_WORLD);
			}

			// Obtenemos del nodo cero la bandera que indica si debemos continuar
			// calculando o si por contra ya se ha llegado al valor de convergencia
			// deseado
			MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Liberamos la memoria utilizada por cada nodo
	if (rank != 0)
	{
		free(C);
		free(xkp1);
		free(xk);
		free(xconv);
		free(T);
	}
	else
	{
		free(xconv);
	}
	
	free(LPU);
	free(Dinv);
	
	return k;
}


