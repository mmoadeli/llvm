
//===-------- matrix-tensorcores.hpp - matrix ext impl ---*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once
#include "matrix-unified-utils.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

} // namespace matrix
} // namespace experimental

namespace detail {

template <typename T, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t Rows, size_t Cols,
          sycl::ext::oneapi::experimental::matrix::layout Layout =
              sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          typename Cond = void>
struct joint_matrix_hip;

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR(TYPE, USE, M, N)                      \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_hip<                                                     \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::USE, M, N, Layout,   \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    TYPE data;                                                                 \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, a, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, b, 4, 16)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N, SIZE)                  \
  template <>                                                                   \
  struct joint_matrix_hip<                                                      \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,    \
      sycl::ext::oneapi::experimental::matrix::layout::dynamic> {               \
    using vType = __attribute__( (__vector_size__(SIZE * sizeof(TYPE)) )) TYPE; \
    vType wi_marray;                                                            \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16, 4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(PRECISION, USE, M, N, TYPE, \
                                                   SIZE)                       \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_hip<                                                     \
      PRECISION, sycl::ext::oneapi::experimental::matrix::use::USE, M, N,      \
      Layout,                                                                  \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    TYPE data;                                                                 \
  };

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION
#if defined(__SYCL_DEVICE_ONLY__) && defined(__HIP_PLATFORM_AMD__)
template <sycl::ext::oneapi::experimental::matrix::layout Layout>
constexpr int get_layout_id();

template <>
constexpr int
get_layout_id<sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
  return 0;
}

template <>
constexpr int
get_layout_id<sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
  return 1;
}

template <sycl::ext::oneapi::experimental::matrix::layout Layout, typename S,
          typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
void load_accumulator_layoutT(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride) {
};

template <typename S, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
void load_accumulator_hip(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
  switch (Layout) {
  case sycl::ext::oneapi::experimental::matrix::layout::row_major:
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::row_major>(res, src,
                                                                    stride);
    break;
  case sycl::ext::oneapi::experimental::matrix::layout::col_major:
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::col_major>(res, src,
                                                                    stride);
    break;
  default:
    assert(false && "Invalid layout specified!");
  }
}

template <
    typename Group,
    typename S, typename T, size_t NumRows, size_t NumCols,
    sycl::ext::oneapi::experimental::matrix::use Use,
    sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    typename = typename std::enable_if_t<
        (Layout == sycl::ext::oneapi::experimental::matrix::layout::row_major || 
        Layout == sycl::ext::oneapi::experimental::matrix::layout::col_major) &&
        (NumRows == 16 || NumRows == 4) && (NumCols == 16 || NumCols == 4) && std::is_same_v<S, float>>>
void load_multiplicand_hip(
    joint_matrix_hip<S, Use, NumRows, NumCols, Layout> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group sg) {
  auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
             sg.get_local_linear_id();

  if constexpr (Layout ==
                sycl::ext::oneapi::experimental::matrix::layout::row_major) {
    res.data = src[idx];
    } else if constexpr (Layout ==
                          sycl::ext::oneapi::experimental::matrix::layout::col_major) {
      res.data = src[(idx % NumRows) * NumCols + idx / NumRows];
    }
}

template <typename Group,
          sycl::ext::oneapi::experimental::matrix::layout Layout, typename T,
          size_t NumRows, size_t NumCols, access::address_space Space,
          access::decorated IsDecorated>
void store_layoutT(
    joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride, Group& sg) {
    auto local_id = sg.get_local_id();
    if constexpr (NumRows == 16 && NumCols == 16) {
      if constexpr (std::is_same_v<T, float>) {
        auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();
        auto thread_x = idx / stride;
        auto thread_y = idx % stride;
        for (int i = 0; i < 4; ++i) {
          dst[thread_y + i * stride + thread_x * 4 * stride] = src.wi_marray[i];
        }
      }
    } else {
      static_assert(false && "Invalid dadimenstions!");
    }
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
void joint_matrix_store_hip(
    joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout,
    Group& sg) {
  switch (Layout) {
  case sycl::ext::oneapi::experimental::matrix::layout::row_major:
    store_layoutT<Group,
                  sycl::ext::oneapi::experimental::matrix::layout::row_major>(
        src, dst, stride, sg);
    break;
  case sycl::ext::oneapi::experimental::matrix::layout::col_major:
    store_layoutT<Group,
                  sycl::ext::oneapi::experimental::matrix::layout::col_major>(
        src, dst, stride, sg);
    break;
  default:
    assert(false && "Invalid layout specified!");
  }
}

template <sycl::ext::oneapi::experimental::matrix::layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::layout LayoutB>
constexpr int get_layout_pair_id();

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::row_major,
    sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
  return 0;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::row_major,
    sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
  return 1;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::col_major,
    sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
  return 2;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::col_major,
    sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
  return 3;
}

template <
    typename Tm, typename Tc, std::size_t M, std::size_t K, std::size_t N,
    sycl::ext::oneapi::experimental::matrix::layout LayoutA,
    sycl::ext::oneapi::experimental::matrix::layout LayoutB,
    std::enable_if_t<
        (LayoutA ==
             sycl::ext::oneapi::experimental::matrix::layout::row_major ||
         LayoutA ==
             sycl::ext::oneapi::experimental::matrix::layout::col_major) &&
            (LayoutB ==
                 sycl::ext::oneapi::experimental::matrix::layout::row_major ||
             LayoutB ==
                 sycl::ext::oneapi::experimental::matrix::layout::col_major),
        bool> = true>
void joint_matrix_mad_hip(
    joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &D,
    joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::a, M, K,
                      LayoutA> &A,
    joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::b, K, N,
                      LayoutB> &B,
    joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &C) {
  if constexpr (M == 16 && N == 16 && K == 4) {
    if constexpr (std::is_same_v<Tc, float>) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_16x16x4f32(A.data, B.data, C.wi_marray, 0, 0, 0);
    }
  } else {
    assert(false && "Invalid dimensions!");
  }
}

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
