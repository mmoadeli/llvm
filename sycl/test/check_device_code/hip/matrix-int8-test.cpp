// REQUIRES: hip

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=amd_gpu_gfx90a -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o out.ll -D__HIP_PLATFORM_AMD__=1 -D__HIPCC__=1

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int stride = 16;

int main() {

  buffer<int8_t, 1> bufA(nullptr, range<1>(1));
  buffer<int8_t, 1> bufB(nullptr, range<1>(1));
  buffer<int32_t, 1> bufC(nullptr, range<1>(1));
  buffer<int32_t, 1> bufD(nullptr, range<1>(1));

  queue q;

  q.submit([&](handler &cgh) {
    sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accB(bufB, cgh);
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accC(bufC, cgh);
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD(bufD, cgh);

    cgh.parallel_for<class row_col_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 16, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 16, layout::col_major>
              sub_b{};

          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
        });

    cgh.parallel_for<class col_row_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 16, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 16, layout::row_major>
              sub_b{};

          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
        });
  });

  return 0;
};
