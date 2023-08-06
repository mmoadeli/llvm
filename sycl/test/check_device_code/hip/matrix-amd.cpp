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

  q.submit([&](handler &cgh) {

    cgh.parallel_for<class row_row>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::a, M, K, layout::row_major>
              sub_a{};
        });
  });

  return 0;
};

