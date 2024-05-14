#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"
#include "spmm_naive.h"
#include "spmm_optimized.h"

// use args to pass the parameters
int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Usage: %s <M> <K> <N> <sparse_density>\n", argv[0]);
        return 1;
    }

    Timer timer;

    // generate a sparse matrix
    int sparse_row = atoi(argv[1]);
    int sparse_col = atoi(argv[2]);
    float sparse_density = atof(argv[4]);
    int dense_row = sparse_col;
    int dense_col = atoi(argv[3]);

    float *sparse_matrix = new float[sparse_row * sparse_col];
    int num_nonzero = generate_sparse_matrix(sparse_row, sparse_col, sparse_density, sparse_matrix);

    // convert the sparse matrix to csr format
    int *row_ptr = new int[sparse_row + 1];
    int *col_idx = new int[num_nonzero];
    float *val = new float[num_nonzero];
    convert_sparse_matrix_to_csr(sparse_row, sparse_col, sparse_matrix, row_ptr, col_idx, val);


    // generate a dense matrix
    float *dense_matrix = new float[dense_row * dense_col];
    generate_dense_matrix(dense_row, dense_col, dense_matrix);

    // run a plain GEMM for checking the correctness
    int m = sparse_row;
    int n = dense_col;
    int k = sparse_col;
    float *A = sparse_matrix;
    float *B = dense_matrix;
    float *C = new float[m * n];
    gemm(m, n, k, A, B, C);

    // run a naive SPMM
    float *res = new float[m * n];
    memset(res, 0, sizeof(float) * m * n);
    spmm_naive(row_ptr, col_idx, val, dense_matrix, res, m, n, k);
    
    compare_matrix(m, n, C, res);

    // run a optimized SPMM
    float *res_optimized = new float[m * n];
    memset(res_optimized, 0, sizeof(float) * m * n);
    spmm_cpu_optimized_no_tile_v1(row_ptr, col_idx, val, dense_matrix, res_optimized, m, n, k);

    compare_matrix(m, n, C, res_optimized);

    // print_matrix(m, n, C);
    // print_matrix(m, n, res);

    delete[] sparse_matrix;
    delete[] row_ptr;
    delete[] col_idx;
    delete[] val;
    delete[] dense_matrix;
    delete[] C;
    delete[] res;

    return 0;
}
