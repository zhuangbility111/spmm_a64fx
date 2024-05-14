#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "utils.h"
#include "spmm_optimized.h"

#ifdef __ARM_FEATURE_SVE /* __ARM_FEATURE_SVE */
#include <arm_sve.h>
#define VEC_LEN 16

using Inner_kernel = void (*)(int*, float*, float*, float*, int, int, int, int, int, svbool_t&, svbool_t&,
                              svbool_t&, svbool_t&);

template <int N>
void kernel_1xN(int* col, float* value, float* mat, float* out, int m, int n, int ldb, int start_on_cols,
                int end_on_cols, svbool_t& pg0, svbool_t& pg1, svbool_t& pg2, svbool_t& pg3) {
    svfloat32_t vout0, vout1, vout2, vout3;
    svfloat32_t va;
    svfloat32_t vb0, vb1, vb2, vb3;
    int out_idx = m * ldb + n;
    // load output to SVE register
    if (N > 0) vout0 = svld1(pg0, &(out[out_idx]));
    if (N > 1) vout1 = svld1(pg1, &(out[out_idx + VEC_LEN]));
    if (N > 2) vout2 = svld1(pg2, &(out[out_idx + 2 * VEC_LEN]));
    if (N > 3) vout3 = svld1(pg3, &(out[out_idx + 3 * VEC_LEN]));

    for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
        int k = col[id_on_cols];
        int b_idx = k * ldb + n;
        // load elem on sparse matrix
        va = svdup_n_f32(value[id_on_cols]);
        // load elems on dense matrix based on the value of N
        if (N > 0) vb0 = svld1(pg0, &(mat[b_idx]));
        if (N > 1) vb1 = svld1(pg1, &(mat[b_idx + VEC_LEN]));
        if (N > 2) vb2 = svld1(pg2, &(mat[b_idx + 2 * VEC_LEN]));
        if (N > 3) vb3 = svld1(pg3, &(mat[b_idx + 3 * VEC_LEN]));

        // fma based on the value of N
        if (N > 0) vout0 = svmla_f32_x(pg0, vout0, va, vb0);
        if (N > 1) vout1 = svmla_f32_x(pg1, vout1, va, vb1);
        if (N > 2) vout2 = svmla_f32_x(pg2, vout2, va, vb2);
        if (N > 3) vout3 = svmla_f32_x(pg3, vout3, va, vb3);
    }

    // store output from SVE register
    if (N > 0) svst1(pg0, &(out[out_idx]), vout0);
    if (N > 1) svst1(pg1, &(out[out_idx + VEC_LEN]), vout1);
    if (N > 2) svst1(pg2, &(out[out_idx + 2 * VEC_LEN]), vout2);
    if (N > 3) svst1(pg3, &(out[out_idx + 3 * VEC_LEN]), vout3);
}

#elif __AVX512F__ /* AVX512 */
#include <immintrin.h>
#define VEC_LEN 16
using Inner_kernel = void (*)(int*, float*, float*, float*, int, int, int, int, int, __mmask16&, __mmask16&,
                              __mmask16&, __mmask16&);

template <int N>
void kernel_1xN(int* col, float* value, float* mat, float* out, int m, int n, int ldb, int start_on_cols,
                int end_on_cols, __mmask16& pg0, __mmask16& pg1, __mmask16& pg2, __mmask16& pg3) {
    __m512 vout0, vout1, vout2, vout3;
    __m512 va;
    __m512 vb0, vb1, vb2, vb3;

    int out_idx = m * ldb + n;

    // load output to SVE register
    if (N == 1) {  // (0, 1]
        vout0 = _mm512_maskz_loadu_ps(pg0, &(out[out_idx]));
    } else if (N == 2) {  // (1, 2]
        vout0 = _mm512_loadu_ps(&(out[out_idx]));
        // vout1 = svld1(pg1, &(out[out_idx + VEC_LEN]));
        vout1 = _mm512_maskz_loadu_ps(pg1, &(out[out_idx + VEC_LEN]));
    } else if (N == 3) {  // (2, 3]
        vout0 = _mm512_loadu_ps(&(out[out_idx]));
        vout1 = _mm512_loadu_ps(&(out[out_idx + VEC_LEN]));
        // vout2 = svld1(pg2, &(out[out_idx + 2 * VEC_LEN]));
        vout2 = _mm512_maskz_loadu_ps(pg2, &(out[out_idx + 2 * VEC_LEN]));
    } else if (N == 4) {  // (3, 4)
        vout0 = _mm512_loadu_ps(&(out[out_idx]));
        vout1 = _mm512_loadu_ps(&(out[out_idx + VEC_LEN]));
        vout2 = _mm512_loadu_ps(&(out[out_idx + 2 * VEC_LEN]));
        // vout3 = svld1(pg3, &(out[out_idx + 3 * VEC_LEN]));
        vout3 = _mm512_maskz_loadu_ps(pg3, &(out[out_idx + 3 * VEC_LEN]));
    }

    for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
        int k = col[id_on_cols];
        int b_idx = k * ldb + n;

        // load elem on sparse matrix
        va = _mm512_set1_ps(value[id_on_cols]);

        // load elems on dense matrix based on the value of N
        if (N == 1) {  // (0, 1]
            vb0 = _mm512_maskz_loadu_ps(pg0, &(mat[b_idx]));
        } else if (N == 2) {  // (1, 2]
            vb0 = _mm512_loadu_ps(&(mat[b_idx]));
            vb1 = _mm512_maskz_loadu_ps(pg1, &(mat[b_idx + VEC_LEN]));
        } else if (N == 3) {  // (2, 3]
            vb0 = _mm512_loadu_ps(&(mat[b_idx]));
            vb1 = _mm512_loadu_ps(&(mat[b_idx + VEC_LEN]));
            vb2 = _mm512_maskz_loadu_ps(pg2, &(mat[b_idx + 2 * VEC_LEN]));
        } else if (N == 4) {  // (3, 4)
            vb0 = _mm512_loadu_ps(&(mat[b_idx]));
            vb1 = _mm512_loadu_ps(&(mat[b_idx + VEC_LEN]));
            vb2 = _mm512_loadu_ps(&(mat[b_idx + 2 * VEC_LEN]));
            vb3 = _mm512_maskz_loadu_ps(pg3, &(mat[b_idx + 3 * VEC_LEN]));
        }

        // fma based on the value of N
        if (N > 0) vout0 = _mm512_fmadd_ps(va, vb0, vout0);
        if (N > 1) vout1 = _mm512_fmadd_ps(va, vb1, vout1);
        if (N > 2) vout2 = _mm512_fmadd_ps(va, vb2, vout2);
        if (N > 3) vout3 = _mm512_fmadd_ps(va, vb3, vout3);
    }

    if (N == 1) {  // (0, 1]
        _mm512_mask_storeu_ps(&(out[out_idx]), pg0, vout0);
    } else if (N == 2) {  // (1, 2]
        _mm512_storeu_ps(&(out[out_idx]), vout0);
        _mm512_mask_storeu_ps(&(out[out_idx + VEC_LEN]), pg1, vout1);
    } else if (N == 3) {  // (2, 3]
        _mm512_storeu_ps(&(out[out_idx]), vout0);
        _mm512_storeu_ps(&(out[out_idx + VEC_LEN]), vout1);
        _mm512_mask_storeu_ps(&(out[out_idx + 2 * VEC_LEN]), pg2, vout2);
    } else if (N == 4) {  // (3, 4)
        _mm512_storeu_ps(&(out[out_idx]), vout0);
        _mm512_storeu_ps(&(out[out_idx + VEC_LEN]), vout1);
        _mm512_storeu_ps(&(out[out_idx + 2 * VEC_LEN]), vout2);
        _mm512_mask_storeu_ps(&(out[out_idx + 3 * VEC_LEN]), pg3, vout3);
    }
}
#endif

Inner_kernel get_kernel_1xN(int n) {
    if (n == 1)
        return kernel_1xN<1>;
    else if (n == 2)
        return kernel_1xN<2>;
    else if (n == 3)
        return kernel_1xN<3>;
    return kernel_1xN<4>;
}

int obtain_tile_rowptr(int* rowptr, int* col, float* values, int rowptr_start, int rowptr_end, int* tile_rowptr,
                       int tile_num, int tile_size) {
    // std::cout << "tile_num: " << tile_num << ", tile_size: " << tile_size << std::endl;
    int tile_rowptr_id = 0;
    int cur_tile_id = 0;
    tile_rowptr[0] = rowptr[rowptr_start];
    for (int i = rowptr_start; i < rowptr_end; i++) {
        int col_id_start = rowptr[i], col_id_end = rowptr[i + 1];
        int cur_col_id = col_id_start;
        int cur_col = col[cur_col_id];
        for (int cur_tile_id = 0; cur_tile_id < tile_num; cur_tile_id++) {
            ++tile_rowptr_id;
            tile_rowptr[tile_rowptr_id] = tile_rowptr[tile_rowptr_id - 1];
            while (cur_col_id < col_id_end && cur_col / tile_size == cur_tile_id) {
                // std::cout << "row: " << i << ", " << "cur_col: " << cur_col << ", tile_id: " << cur_col / tile_size
                // << ", cur_tile_id: " << cur_tile_id << std::endl;
                tile_rowptr[tile_rowptr_id]++;
                ++cur_col_id;
                cur_col = col[cur_col_id];
            }
        }
    }
    return 1;
}

#ifdef __ARM_FEATURE_SVE /* __ARM_FEATURE_SVE */
Inner_kernel select_kernel(const int N, int& step, svbool_t& pg0, svbool_t& pg1, svbool_t& pg2, svbool_t& pg3) {
    Inner_kernel kernel = nullptr;
    if (N > 3 * VEC_LEN) {
        kernel = get_kernel_1xN(4);
        pg3 = svwhilelt_b32(3 * VEC_LEN, N);
        step = std::min(N, 4 * VEC_LEN);
    } else if (N > 2 * VEC_LEN) {
        kernel = get_kernel_1xN(3);
        pg2 = svwhilelt_b32(2 * VEC_LEN, N);
        step = N;
    } else if (N > 1 * VEC_LEN) {
        kernel = get_kernel_1xN(2);
        pg1 = svwhilelt_b32(1 * VEC_LEN, N);
        step = N;
    } else if (N > 0) {
        kernel = get_kernel_1xN(1);
        pg0 = svwhilelt_b32(0 * VEC_LEN, N);
        step = N;
    }
    return kernel;
}
#elif __AVX512F__ /* AVX512 */
Inner_kernel select_kernel(const int N, int& step, __mmask16& pg0, __mmask16& pg1, __mmask16& pg2, __mmask16& pg3) {
    Inner_kernel kernel = nullptr;
    if (N > 4 * VEC_LEN) {  // (4 * VEC_LEN, +inf)
        kernel = get_kernel_1xN(4);
        step = 4 * VEC_LEN;
    } else if (N > 3 * VEC_LEN && N <= 4 * VEC_LEN) {  // (3 * VEC_LEN, 4 * VEC_LEN]
        kernel = get_kernel_1xN(4);
        pg3 = static_cast<__mmask16>((1 << (N - 3 * VEC_LEN)) - 1);
        step = N;
    } else if (N > 2 * VEC_LEN && N <= 3 * VEC_LEN) {  // (2 * VEC_LEN, 3 * VEC_LEN]
        kernel = get_kernel_1xN(3);
        pg2 = static_cast<__mmask16>((1 << (N - 2 * VEC_LEN)) - 1);
        step = N;
    } else if (N > 1 * VEC_LEN && N <= 2 * VEC_LEN) {  // (1 * VEC_LEN, 2 * VEC_LEN]
        kernel = get_kernel_1xN(2);
        pg1 = static_cast<__mmask16>((1 << (N - 1 * VEC_LEN)) - 1);
        step = N;
    } else if (N > 0 && N <= 1 * VEC_LEN) {  // (0, 1 * VEC_LEN]
        kernel = get_kernel_1xN(1);
        pg0 = static_cast<__mmask16>((1 << N) - 1);
        step = N;
    }
    return kernel;
}
#endif

// std::tuple<torch::Tensor, torch::optional<torch::Tensor>> spmm_cpu_optimized_no_tile_v1(
void spmm_cpu_optimized_no_tile_v1(int* rowptr, int* col, float* value, float* mat, float* out, int M, int N, int K) {
    int* rowptr_data = rowptr;
    int* col_data = col;
    float* value_data = value;
    float* mat_data = mat;
    float* out_data = out;

    int max_num_threads = omp_get_max_threads();
    int num_threads_on_vertexs = max_num_threads;
    int num_threads_on_features = 1;

    // divide work
    int* work_range_on_vertexs = nullptr;
    int* work_range_on_features = nullptr;

    // if the work partition is pre-defined, we can reuse it
    // else divide the work by the number of threads directly.
    if (work_range_on_vertexs == nullptr) {
        printf("divide row work dynamically\n");
        work_range_on_vertexs = new int[num_threads_on_vertexs + 1];
        divide_work(work_range_on_vertexs, M, num_threads_on_vertexs);
    }

    if (work_range_on_features == nullptr) {
        printf("divide col work dynamically\n");
        work_range_on_features = new int[num_threads_on_features + 1];
        divide_work(work_range_on_features, N, num_threads_on_features);
    }

    printf("vector length: %d\n", svcntw());

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int tid_on_vertexs = tid / num_threads_on_features;
        int tid_on_features = tid % num_threads_on_features;

        int start_on_M = work_range_on_vertexs[tid_on_vertexs];
        int end_on_M = work_range_on_vertexs[tid_on_vertexs + 1];

        int start_on_N = work_range_on_features[tid_on_features];
        int end_on_N = work_range_on_features[tid_on_features + 1];
        int step_on_N = end_on_N - start_on_N;

#ifdef __ARM_FEATURE_SVE /* __ARM_FEATURE_SVE */
        svbool_t pg0_main = svptrue_b32();
        svbool_t pg1_main = svptrue_b32();
        svbool_t pg2_main = svptrue_b32();
        svbool_t pg3_main = svptrue_b32();

        svbool_t pg0_corner = svptrue_b32();
        svbool_t pg1_corner = svptrue_b32();
        svbool_t pg2_corner = svptrue_b32();
        svbool_t pg3_corner = svptrue_b32();
#elif __AVX512F__ /* AVX512 */
        // set mask register to all true
        __mmask16 pg0_main = static_cast<__mmask16>(0xFFFF);
        __mmask16 pg1_main = static_cast<__mmask16>(0xFFFF);
        __mmask16 pg2_main = static_cast<__mmask16>(0xFFFF);
        __mmask16 pg3_main = static_cast<__mmask16>(0xFFFF);

        __mmask16 pg0_corner = static_cast<__mmask16>(0xFFFF);
        __mmask16 pg1_corner = static_cast<__mmask16>(0xFFFF);
        __mmask16 pg2_corner = static_cast<__mmask16>(0xFFFF);
        __mmask16 pg3_corner = static_cast<__mmask16>(0xFFFF);
#endif

        // select kernel
        int step_main_kernel = 0, step_corner_kernel = 0;
        Inner_kernel main_kernel =
            select_kernel(step_on_N, step_main_kernel, pg0_main, pg1_main, pg2_main, pg3_main);
        Inner_kernel corner_kernel = select_kernel(step_on_N % step_main_kernel, step_corner_kernel, pg0_corner,
                                                    pg1_corner, pg2_corner, pg3_corner);
        int start_on_N_main = start_on_N;
        int end_on_N_main = end_on_N - step_corner_kernel;
        int start_on_N_corner = end_on_N_main;
        int end_on_N_corner = end_on_N;

        for (int m = start_on_M; m < end_on_M; m++) {
            int start_on_cols = rowptr_data[m];
            int end_on_cols = rowptr_data[m + 1];
            for (int n = start_on_N_main; n < end_on_N_main; n += step_main_kernel) {
                main_kernel(col_data, value_data, mat_data, out_data, m, n, N, start_on_cols, end_on_cols, pg0_main,
                            pg1_main, pg2_main, pg3_main);
            }

            if (start_on_N_corner < end_on_N_corner) {
                corner_kernel(col_data, value_data, mat_data, out_data, m, start_on_N_corner, N, start_on_cols,
                                end_on_cols, pg0_corner, pg1_corner, pg2_corner, pg3_corner);
            }
        }
    }

}
