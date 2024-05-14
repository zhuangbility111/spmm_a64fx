int generate_sparse_matrix(int row, int col, float density, float *matrix);

void convert_sparse_matrix_to_csr(int row, int col, float *matrix, int *row_ptr, int *col_idx, float *val);

void generate_dense_matrix(int row, int col, float *matrix);

void compare_matrix(int m, int n, float *A, float *B);

void print_matrix(int m, int n, float *matrix);

int divup(int x, int y);

void divide_work(int* work_range, int total_work, int num_threads);