// REQUIRES: hip

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=amd_gpu_gfx90a -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o out.ll -D__HIP_PLATFORM_AMD__=1  -I/opt/rocm-5.4.3/include
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// M, N, K define the sizes of dimensions of the three matrix types (a, b,
// accumulator) used per subgroup operation.
constexpr int M = 16; // number of rows of accumulator,
                     // number of cols of b.
constexpr int N = 8; // number of cols of accumulator,
                     // number of rows of a.
constexpr int K = 16; // number of cols of a/number of rows of b.


int main() {

  queue q;
  buffer<int32_t, 1> bufA(nullptr, range<1>(1));
  buffer<int32_t, 1> bufB(nullptr, range<1>(1));
  buffer<int32_t, 1> bufC(nullptr, range<1>(1));
  buffer<int32_t, 1> bufD(nullptr, range<1>(1));

  size_t stride = 8;

  q.submit([&](handler &cgh) {
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accB(bufB, cgh);
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accC(bufC, cgh);
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD(bufD, cgh);
    cgh.parallel_for<class row_row>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::a, M, K, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, int32_t, use::b, M, K, layout::col_major>
              sub_b{};

          joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};

          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);

          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);

          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
        });
  });

  return 0;
};