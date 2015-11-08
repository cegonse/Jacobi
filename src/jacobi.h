#ifndef _JACOBI_H_
#define _JACOBI_H_

#include <math.h>
#include <malloc.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <signal.h>
#include <omp.h>
#include <mkl.h>
#include <string.h>

#include ".math/matsub.h"

// Errores devueltos por el algoritmo de Jacobi:
// 
// > ENONDOM: la matriz de términos independientes
// 		      del sistema de ecuaciones no es de
//	          diagonal estrictamente dominante, por
//            lo que no está asegurada la convergencia.
enum
{
	ENONDOM = -1
};

// Iteración de Jacobi
// - Esta función ejecuta la iteración de Jacobi hasta
//   que la solución obtenida tenga una cota de error
//   inferior a la especificada.
//
// - En función del tamaño del problema, utilizará nucleos
//   computacionales paralelos o no.
//
// Parámetros de entrada:
// > (double[][]) A: matriz de términos independientes del sistema
//    		         de ecuaciones.
//
// > (double[]) B: vector de coeficientes del sistema de ecuaciones.
//
// > (double[]) x0: vector con la aproximación inicial a la solución
//     		        del sistema.
//
// > (double) conv: valor de error mínimo para el que debe de converger
// 				    el sistema.
//
// > (int) n: tamaño del lado de la matriz A y longitud de los vectores b
//			  y x0.
//
// Parámetros de salida:
// > (int): devuelve el número de iteraciones que han sido necesarias para
//	        obtener la solución, o un error en caso de fallo.
int jacobi(double *A, double *b, double *x0, double conv, int n);


void getct(double *Dinv, double *LPU, double *T, double *b, double *C, int n);
static inline void __getct_kernel_sequential(double *Dinv, double *LPU, double *T, double *b, double *C, int n);
static inline void __getct_kernel_parallel(double *Dinv, double *LPU, double *T, double *b, double *C, int n, int chunk, int p);


double* getlpu(double *A, int n, double *LPU);
static inline void __getlpu_kernel_sequential(double *A, int n, double *LPU);
static inline void __getlpu_kernel_parallel(double *A, int n, double *LPU, int p);


int isdom(double *A, int n);
static inline int __isdom_kernel_sequential(double *A, int n);
static inline int __isdom_kernel_parallel(double *A, int n, int p);


double* diaginv(double *A, int n, double *diag);
static inline void __diaginv_kernel_sequential(double *A, int n, double *diag);
static inline void __diaginv_kernel_parallel(double *A, int n, double *diag, int p);


void printMat(double* A, int n);
void printVec(double* a, int n);


// Declaraciones de funciones del BLAS
int dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, 
	double *b, int *ldb, double *beta, double *c, int *ldc);

double dnrm2_(int *n, double *x, int *incx);

#endif // _JACOBI_H_
