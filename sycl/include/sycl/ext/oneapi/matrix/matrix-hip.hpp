
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

#if defined(__SYCL_DEVICE_ONLY__)

#define __SYCL_JOINT_MATRIX_HALF_OVERLOAD_ARR(USE, M, N, SIZE)                 \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_hip<                                                     \
      sycl::half, USE, M, N, Layout,                                           \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    using vType =                                                              \
        __attribute__((__vector_size__(SIZE * sizeof(__fp16)))) __fp16;        \
    vType data = {0};                                                          \
  };

__SYCL_JOINT_MATRIX_HALF_OVERLOAD_ARR(
    sycl::ext::oneapi::experimental::matrix::use::a, 16, 4, 4)
__SYCL_JOINT_MATRIX_HALF_OVERLOAD_ARR(
    sycl::ext::oneapi::experimental::matrix::use::b, 4, 16, 4)
__SYCL_JOINT_MATRIX_HALF_OVERLOAD_ARR(
    sycl::ext::oneapi::experimental::matrix::use::a, 32, 8, 4)
__SYCL_JOINT_MATRIX_HALF_OVERLOAD_ARR(
    sycl::ext::oneapi::experimental::matrix::use::b, 8, 32, 4)
__SYCL_JOINT_MATRIX_HALF_OVERLOAD_ARR(
    sycl::ext::oneapi::experimental::matrix::use::a, 32, 4, 4)
__SYCL_JOINT_MATRIX_HALF_OVERLOAD_ARR(
    sycl::ext::oneapi::experimental::matrix::use::b, 4, 32, 4)

#endif

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N, SIZE)                 \
  template <>                                                                  \
  struct joint_matrix_hip<                                                     \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,   \
      sycl::ext::oneapi::experimental::matrix::layout::dynamic> {              \
    using vType = __attribute__((__vector_size__(SIZE * sizeof(TYPE)))) TYPE;  \
    vType wi_marray = {0};                                                     \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32, 16)

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
          access::address_space Space, access::decorated IsDecorated,
          typename Group>
void load_accumulator_layoutT(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group &sg) {
  auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
             sg.get_local_linear_id();
  if constexpr (Layout ==
                sycl::ext::oneapi::experimental::matrix::layout::row_major) {
    res.wi_marray = {0};
  } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                     layout::col_major) {
    res.wi_marray = {0};
  }
};

template <typename Group, typename S, typename T, size_t NumRows,
          size_t NumCols, access::address_space Space,
          access::decorated IsDecorated>
void load_accumulator_hip(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout, Group &sg) {
  switch (Layout) {
  case sycl::ext::oneapi::experimental::matrix::layout::row_major:
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::row_major>(res, src,
                                                                    stride, sg);
    break;
  case sycl::ext::oneapi::experimental::matrix::layout::col_major:
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::col_major>(res, src,
                                                                    stride, sg);
    break;
  default:
    assert(false && "Invalid layout specified!");
  }
}

template <
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols,
    sycl::ext::oneapi::experimental::matrix::use Use,
    sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    typename = typename std::enable_if_t<
        (Layout == sycl::ext::oneapi::experimental::matrix::layout::row_major ||
         Layout == sycl::ext::oneapi::experimental::matrix::layout::col_major)>>
void load_multiplicand_hip(
    joint_matrix_hip<S, Use, NumRows, NumCols, Layout> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group &sg) {
  auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
             sg.get_local_linear_id();
  if constexpr (NumRows == 16 || NumRows == 4)
    &&(NumCols == 16 || NumCols == 4) {
      if constexpr (std::is_same_v<S, float>) {
        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major) {
          res.data = src[idx];
        } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                           layout::col_major) {
          res.data = src[(idx % NumRows) * NumCols + idx / NumRows];
        }
      } else if constexpr (std::is_same_v<S, half>) {
        auto thread_x = idx % 16;
        auto thread_y = idx / 16;

        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major) {
          for (int i = 0; i < 4; ++i) {
            const int r_idx = thread_x * 4 + i + thread_y * 64;
            res.data[i] = src[r_idx];
          }
        } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                           layout::col_major) {
          for (int i = 0; i < 4; ++i) {
            const int c_idx =
                thread_x + 4 * i + thread_y * 64 + (thread_x / 4) * 12;
            res.data[i] = src[c_idx];
          }
        }
      }
    }
  else if constexpr (NumRows == 32 || NumRows == 8)
    &&(NumCols == 32 || NumCols == 8) {
      if constexpr (std::is_same_v<S, half>) {
        auto thread_x = idx % 32;
        auto thread_y = idx / 32;

        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major) {
          for (int i = 0; i < 4; ++i) {
            const int r_idx = thread_x * 8 + i + thread_y * 4;
            res.data[i] = src[r_idx];
          }
        } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                           layout::col_major) {
          for (int i = 0; i < 4; ++i) {
            const int c_idx = thread_x + i * 32 + thread_y * 4 * 32;
            res.data[i] = src[c_idx];
          }
        }
      }
    }
  else if constexpr (NumRows == 32 || NumRows == 4)
    &&(NumCols == 32 || NumCols == 4) {
      if constexpr (std::is_same_v<S, half>) {
        auto thread_x = idx % 32;
        auto thread_y = idx / 32;

        constexpr int LDA = 4;
        constexpr int LDB = 32;
        constexpr int batchStrideA = M * LDA;
        constexpr int batchStrideB = K * LDB;

        if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                    layout::row_major) {
          for (int i = 0; i < 4; ++i) {
            const int r_idx = thread_x * LDA + i + thread_y * batchStrideA;
            res.data[i] = src[r_idx];
          }
        } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                           layout::col_major) {
          for (int i = 0; i < 4; ++i) {
            const int c_idx = thread_x + i * LDB + thread_y * batchStrideB;
            res.data[i] = src[c_idx];
          }
        }
      }
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
    multi_ptr<T, Space, IsDecorated> dst, size_t stride, Group &sg) {
  if constexpr (NumRows == 16 && NumCols == 16) {
    if constexpr (std::is_same_v<T, float>) {
      auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                 sg.get_local_linear_id();
      auto thread_x = idx % stride;
      auto thread_y = idx / stride;

      for (int b = 0; b < 4; ++b) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = thread_x * 16 + i + thread_y * 4 + b * 16;
          dst[d_idx] = src.wi_marray[i + b * 4];
        }
      }
    }
  } else if constexpr (NumRows == 32 && NumCols == 32) {
    if constexpr (std::is_same_v<T, float>) {
      auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                 sg.get_local_linear_id();
      auto thread_x = idx % stride;
      auto thread_y = idx / stride;
      const int LDD = 32;

      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx =
              thread_x + i * LDD + thread_y * 4 * LDD + j * 2 * 4 * LDD;
          dst[d_idx] = src.wi_marray[i + 4 * j];
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
    if constexpr (std::is_same_v<Tm, float>) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_16x16x4f32(A.data, B.data, C.wi_marray, 0, 0, 0);
    } else if constexpr (std::is_same_v<Tm, sycl::half>) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_16x16x4f16(A.data, B.data, C.wi_marray, 0, 0, 0);
    }
  } else if constexpr (M == 32 && N == 32 && K == 8)
    if constexpr (std::is_same_v<Tm, sycl::half>) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_32x32x8f16(A.data, B.data,
                                                         C.wi_marray, 0, 0, 0);
    }
}
else if constexpr (M == 32 && N == 32 &&
                   K == 4) if constexpr (std::is_same_v<Tm, sycl::half>) {
  D.wi_marray = __builtin_amdgcn_mfma_f32_32x32x4f16(A.data, B.data,
                                                     C.wi_marray, 0, 0, 0);
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
