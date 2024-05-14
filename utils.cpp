#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// generate left matrix, a sparse matrix
// parameters: row, col, density, use a float array to store the matrix
int generate_sparse_matrix(int row, int col, float density, float *matrix) {
    int num_nonzero = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if ((float)rand() / RAND_MAX < density) {
                matrix[i * col + j] = (float)rand() / RAND_MAX;
                num_nonzero++;
            } else {
                matrix[i * col + j] = 0;
            }
        }
    }
    return num_nonzero;
}

// convert the sparse matrix to csr format
// parameters: row, col, matrix, row_ptr, col_idx, val
void convert_sparse_matrix_to_csr(int row, int col, float *matrix, int *row_ptr, int *col_idx, float *val) {
    int count = 0;
    for (int i = 0; i < row; i++) {
        row_ptr[i] = count;
        for (int j = 0; j < col; j++) {
            if (matrix[i * col + j] != 0.0) {
                col_idx[count] = j;
                val[count] = matrix[i * col + j];
                count++;
            }
        }
    }
    row_ptr[row] = count;
}

// generate a dense matrix
// parameters: row, col, use a float array to store the matrix
void generate_dense_matrix(int row, int col, float *matrix) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            // generate a random float number between 0 and 1
            matrix[i * col + j] = (float)rand() / RAND_MAX;
        }
    }
}

void compare_matrix(int m, int n, float *A, float *B) {
    int nonzero_idx = -1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i * n + j] != 0.0 && nonzero_idx == -1) {
                nonzero_idx = i * n + j;
            }
            if (abs(A[i * n + j] - B[i * n + j]) / abs(B[i * n + j]) > 1e-6) {
                printf("Error: A[%d][%d] = %f, B[%d][%d] = %f\n", i, j, A[i * n + j], i, j, B[i * n + j]);
                return;
            }
        }
    }
	if (nonzero_idx == -1)
		printf("Correct! but the result is a 0 matrix.\n");
	else
    	printf("Correct! A[first_nonzero_idx] = %f, B[first_nonzero_idx] = %f\n", A[nonzero_idx], B[nonzero_idx]);
}

void print_matrix(int m, int n, float *matrix) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("--------------------\n");
}

int divup(int x, int y) { return (x + y - 1) / y; }

void divide_work(int* work_range, int total_work, int num_threads) {
    int chunk_size;
    int remain_work = total_work;
    work_range[0] = 0;
    for (int i = 0; i < num_threads; i++) {
        chunk_size = divup(remain_work, num_threads - i);
        work_range[i + 1] = work_range[i] + chunk_size;
        remain_work -= chunk_size;
    }
    work_range[num_threads] = total_work;
}