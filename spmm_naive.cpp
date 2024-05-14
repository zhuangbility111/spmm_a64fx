#include <omp.h>

void gemm(int m, int n, int k, float *A, float *B, float *C) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void spmm_naive(int* row_ptr, int* col_idx, float* val, float* dense_matrix, float* res, int m, int n, int k) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        for (int e = row_start; e < row_end; e++) {
            int c = col_idx[e];
            for (int j = 0; j < n; j++) {
                res[i * n + j] += val[e] * dense_matrix[c * n + j];
            }
        }
    }
}