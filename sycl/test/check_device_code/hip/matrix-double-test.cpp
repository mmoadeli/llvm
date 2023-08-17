// REQUIRES: cuda

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=amd_gpu_gfx90a -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o out.ll -D__HIP_PLATFORM_AMD__=1 -D__HIPCC__=1

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// M, N, K define the sizes of dimensions of the three matrix types (a, b,
// accumulator) used per subgroup operation.
constexpr int M = 16; // number of rows of accumulator,
                     // number of cols of b.
constexpr int N = 16; // number of cols of accumulator,
                     // number of rows of a.
constexpr int K = 16; // number of cols of a/number of rows of b.

double A[M * K];
double B[K * N];
double C[M * N];
double D[M * N];

int main() {

  buffer<double, 1> bufA(A, range<1>(M * K));
  buffer<double, 1> bufB(B, range<1>(K * N));
  buffer<double, 1> bufC(C, range<1>(M * N));
  buffer<double, 1> bufD(D, range<1>(M * N));

  queue q;

  q.submit([&](handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accB(bufB, cgh);
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accC(bufC, cgh);
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD(bufD, cgh);

    cgh.parallel_for<class row_col>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, double, use::accumulator, M, N> sub_c{};
          joint_matrix<sub_group, double, use::a, M, K, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, double, use::b, K, N, layout::col_major>
              sub_b{};

          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              N, layout::row_major);
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              K);
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              N);
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              N, layout::row_major);
        });

    cgh.parallel_for<class col_row>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, double, use::accumulator, M, N> sub_c{};
          joint_matrix<sub_group, double, use::a, M, K, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, double, use::b, K, N, layout::row_major>
              sub_b{};

          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              M, layout::col_major);
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              M);
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              K);
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              M, layout::col_major);
        });
  });

  return 0;
};
