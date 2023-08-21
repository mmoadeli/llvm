// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx90a -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 %s -o matrix.out  -D__HIP_PLATFORM_AMD__=1  -D__HIPCC__=1 -D__gfx90a__
// RUN: %{run} %matrix.out


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

using DataType = float;

DataType A[M * K];
DataType B[K * N];
DataType C[M * N];
DataType D[M * N];

int main() {

  for (auto i = 0; i < M; ++i)
    for (auto j = 0; j < K; ++j) {
        A[i * M + j] = 1;
        B[i * M + j] = 1;
        C[i * M + j] = 0;
        D[i * M + j] = 0;
    }

  buffer<DataType, 1> bufA(A, range<1>(M * K));
  buffer<DataType, 1> bufB(B, range<1>(K * N));
  buffer<DataType, 1> bufC(C, range<1>(M * N));
  buffer<DataType, 1> bufD(D, range<1>(M * N));

  queue q;

  q.submit([&](handler &cgh) {
    sycl::accessor<DataType, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<DataType, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accB(bufB, cgh);
    sycl::accessor<DataType, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accC(bufC, cgh);
    sycl::accessor<DataType, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD(bufD, cgh);

    cgh.parallel_for<class row_col>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, DataType, use::accumulator, M, N, layout::row_major> sub_c{};
          joint_matrix<sub_group, DataType, use::a, M, K, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, DataType, use::b, K, N, layout::col_major>
              sub_b{};

          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              N);
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
  }).wait();

  assert(D[0] == 16);
  return 0;
};
