void gemm(int m, int n, int k, float *A, float *B, float *C);

void spmm_naive(int* row_ptr, int* col_idx, float* val, float* dense_matrix, float* res, int m, int n, int k);