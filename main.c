
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "memory.h"
#include "sys/time.h"
#include "omp.h"

#include "Lab3IO"
#include "timer.h"

/* Thread count. */
int threads;

/* Index a two dimensional matrix.
 * Precondition: N is in scope and is the stride of the matrix */
#define INX(arr, row, col) (arr[(row)*N + (col)])

#define STRINGIFY(exp) #exp
#define TOSTRING(x) STRINGIFY(x)

/* Single threaded gaussian elimination */
void gaussian_eliminate_single(float *const Au, const int n) {
	/* Space to use when swapping rows */
	const float swapSize = sizeof(float)*(n+1);
	float *const swapSpace = malloc(swapSize);

	/* Stride of the matrix */
	const int N = n+1;

	/* We set the num threads here at run-time so that our instrumentation
	 * can run a test with varying thread counts without recompilling */
	omp_set_num_threads(threads);

#pragma omp parallel
	/* Main gaussian elimination loop. n-1 bound as the n'th iteration
	 * would have no effect (there are no following rows to eliminate
	 * from using it) */
	for (int k = 0; k < (n-1); ++k) {
#pragma omp single
		{
			/* Locate the row with the largest leading cooeficient */
			int maxRow = k;
			float maxVal = fabs(INX(Au, k, k));
			for (int row = k; row < n; ++row) {
				float sz = fabs(INX(Au, row, k));
				if (sz > maxVal) {
					maxRow = row;
					maxVal = sz;
				}
			}

			/* Swap the max row with the first (but only if a swap is needed) */
			if (maxRow != k) {
				memcpy(swapSpace,     &Au[k*N],      swapSize);
				memcpy(&Au[k*N],      &Au[maxRow*N], swapSize);
				memcpy(&Au[maxRow*N], swapSpace,     swapSize);
			}
		}

		/* Eliminate selected k'th row from following rows */
#pragma omp for
		for (int i = k+1; i < n; ++i) {
			float factor = INX(Au, i, k) / INX(Au, k, k);
			for (int j = k; j <= n; ++j) {
				INX(Au, i, j) -= factor * INX(Au, k, j);
			}
		}
	}
}

/* Single threaded jordinian elimination */
void jordinian_eliminate_single(float *const Ag, const int n) {
	/* Stride of the matrix */
	const int N = n+1;

	/* Main jordinian elimination loop */
	/* Note: Parallelization may not be worth it here, as the jordinian
	 * elimination step is only an tiny portion of the work (O(N^2) as opposed
	 * to O(N^3) like the gaussian step) */
#pragma omp parallel
	for (int k = n-1; k > 0; --k) {
#pragma omp for
		for (int i = 0; i < k; ++i) {
			/* Modify the solution value */
			INX(Ag, i, n) -= (INX(Ag, i, k) / INX(Ag, k, k)) * INX(Ag, k, n);

			/* Zero the coefficient, we eliminated it.
			 * Included for debugging, but not strictly necessary, since we 
			 * don't read from that index or depend on it in subsequent code. 
			 */
			INX(Ag, i, k) = 0;
		}
	}
}

/* Extract solution values from a simplified diagonal+augmented matrix into
 * a vector of the same size. */
void extract_result(float *Ad, float *x, int n) {
	/* Stride of the matrix */
	const int N = n+1;

	/* Get result */
	for (int i = 0; i < n; ++i) {
		x[i] = INX(Ad, i, n) / INX(Ad, i, i);
	}
}

int main(int argc, char *argv[]) {
	/* Usage check */
	if (argc != 3) {
		printf("Usage: main MATSIZE THREADCOUNT\n");
		return EXIT_FAILURE;
	}

	/* Get problem size */
	int n;
	if (sscanf(argv[1], "%d", &n) != 1) {
		printf("Bad matrix size `%s'\n", argv[1]);
		return EXIT_FAILURE;
	}

	/* Get thread count */
	if (sscanf(argv[2], "%d", &threads) != 1) {
		printf("Bad thread count `%s'\n", argv[2]);
		return EXIT_FAILURE;
	}

	/* Read in data */
	float *Au = calloc(n*(n+1), sizeof(float));
	if (LoadLab3(n, Au, 0) != 0) {
		free(Au);
		return EXIT_FAILURE;
	}

	/* Start timing */
	double start_time;
	GET_TIME(start_time);

	/* Run the code */
	gaussian_eliminate_single(Au, n);
	jordinian_eliminate_single(Au, n);

	/* Extract the result */
	float *x = calloc(n, sizeof(float));
	extract_result(Au, x, n);

	/* Done timing */
	double end_time;
	GET_TIME(end_time);
	printf("Calculation with %d threads on %dx%d matrix took %fs.\n", 
		threads,
		n, n,
		(end_time - start_time));

	/* Output */
	SaveLab3(n, x);

	/* Free the things we allocated */
	free(Au);
	free(x);

	/* Done */
	return EXIT_SUCCESS;
}