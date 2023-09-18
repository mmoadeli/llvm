
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
          size_t Rows, size_t Cols, size_t K,
          sycl::ext::oneapi::experimental::matrix::layout Layout =
              sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          typename Cond = void>
struct joint_matrix_hip;

#if defined(__SYCL_DEVICE_ONLY__)

template<typename T>
struct to_hip_type {
  using type = T;
};

template<>
struct to_hip_type<bfloat16> {
  using type = __bf16;
};

template<>
struct to_hip_type<half> {
  using type = __fp16;
};

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR(TYPE, USE, M, N, K, SIZE)             \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_hip<                                                     \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::USE, M, N,           \
      K, Layout,                                                               \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    using vType =                                                              \
        __attribute__((__vector_size__(SIZE * sizeof(typename to_hip_type<TYPE>::type)))) typename to_hip_type<TYPE>::type; \
    vType data;                                                                \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 4, 4, 2, 2)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 4, 4, 2, 2)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 16, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 16, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 32, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 32, 32, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 32, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 32, 2, 2)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 32, 32, 2, 2)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 16, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 16, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 32, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 32, 32, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 32, 32, 8, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 4, 4, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 4, 4, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 16, 16, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 16, 16, 4, 1)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 32, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 32, 32, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 16, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 16, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 32, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 16, 16, 4)


#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N, K, SIZE)               \
  template <>                                                                   \
  struct joint_matrix_hip<                                                      \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N, K, \
      sycl::ext::oneapi::experimental::matrix::layout::dynamic> {               \
    using vType = __attribute__((__vector_size__(SIZE * sizeof(TYPE)))) TYPE;  \
    vType wi_marray = {0};                                                     \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16, 1, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16, 4, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32, 2, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32, 4, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32, 8, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 4, 4, 2, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(double, 4, 4, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(double, 16, 16, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 32, 32, 4, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 16, 16, 4, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 4, 4, 4, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 32, 32, 8, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 16, 16, 16, 4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC

#endif

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
          typename T, size_t NumRows, size_t NumCols, size_t K,
          access::address_space Space, access::decorated IsDecorated,
          typename Group>
void load_accumulator_layoutT(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, K, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
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
          size_t NumCols, size_t K, access::address_space Space,
          access::decorated IsDecorated>
void load_accumulator_hip(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, K, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
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
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols, size_t K,
    sycl::ext::oneapi::experimental::matrix::use Use,
    sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    typename = typename std::enable_if_t<
        (Layout == sycl::ext::oneapi::experimental::matrix::layout::row_major ||
         Layout == sycl::ext::oneapi::experimental::matrix::layout::col_major)>>
void load_multiplicand_hip(
    joint_matrix_hip<S, Use, NumRows, NumCols, K, Layout> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group &sg) {
  auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
             sg.get_local_linear_id();
  
  constexpr bool fp16_bf16_matching_dimensions =
      (std::is_same_v<S, __fp16> || std::is_same_v<S, __bf16>) &&
      ((NumRows == 16 && NumCols == 16 && K == 4) ||
      (NumRows == 16 && NumCols == 16 && K == 16) ||
      (NumRows == 32 && NumCols == 32 && K == 4) ||
      (NumRows == 32 && NumCols == 32 && K == 8) ||
      (NumRows == 4 && NumCols == 4 && K == 4));

  if constexpr (fp16_bf16_matching_dimensions) {
    if constexpr (NumRows == 16 && NumCols == 16 && K == 4) {
      auto thread_x = idx % 16;
      auto thread_y = idx / 16;
      constexpr int batchStrideA = NumRows * K;
      constexpr int batchStrideB = K * NumCols;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                  layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x * K + i + thread_y * batchStrideA;
          res.data[i] = src[r_idx];
        }
      } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                          layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx =
              thread_x + NumCols * i + thread_y * batchStrideB;
          res.data[i] = src[c_idx];
        }
      }
    } else if constexpr (NumRows == 16 && NumCols == 16 && K == 16) {

      auto thread_x = idx % 16;
      auto thread_y = idx / 16;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                  layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x * K + i + thread_y * 4;
          res.data[i] = src[r_idx];
        }
      } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                          layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx = thread_x + i * NumCols + thread_y * NumCols * 4;
          res.data[i] = src[c_idx];
        }
      }
    } else if constexpr (NumRows == 32 && NumCols == 32 && K == 8) {
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
          const int c_idx = thread_x + i * NumCols + thread_y * 4 * NumCols;
          res.data[i] = src[c_idx];
        }
      }
    } else if constexpr (NumRows == 32 && NumCols == 32 && K == 4) {
      auto thread_x = idx % 32;
      auto thread_y = idx / 32;
      constexpr int batchStrideA = NumRows * K;
      constexpr int batchStrideB = K * NumCols;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                  layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x * K + i + thread_y * batchStrideA;
          res.data[i] = src[r_idx];
        }
      } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                          layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx = thread_x + i * NumCols + thread_y * batchStrideB;
          res.data[i] = src[c_idx];
        }
      }
    } else if constexpr (NumRows == 4 && NumCols == 4 && K == 4) {
      auto thread_x = idx % 4;
      auto thread_y = idx / 4;
      constexpr int batchStrideA = 4 * K;
      constexpr int batchStrideB = 4 * NumCols;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                  layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x * K + i + thread_y * batchStrideA;
          res.data[i] = src[r_idx];
        }
      } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                          layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx = thread_x + i * NumCols + thread_y * batchStrideB;
          res.data[i] = src[c_idx];
        }
      }
    } else {
      static_assert(false && "Invalid load dimensions!");
    }
  } else if constexpr (std::is_same_v<S, double>) {
    if constexpr (NumRows == 4 && NumCols == 4 && K == 4) {
      auto thread_x = idx / 4;
      auto thread_y = idx % 4;
      auto thread_xw = thread_x / 4;
      auto thread_xz = thread_x % 4;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                  layout::row_major) {
        const int r_idx = thread_xw * K * NumRows + thread_xz + K * thread_y;
        res.data[0] = src[r_idx];        
      } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                          layout::col_major) {
        const int c_idx = thread_xz * NumCols +  thread_xw * K * NumCols  + thread_y;
        res.data[0] = src[c_idx];
      }
    } else if constexpr (NumRows == 16 && NumCols == 16 && K == 4) {
      auto thread_x = idx / 4;
      auto thread_y = idx % 4;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                  layout::row_major) {
        const int r_idx = K * thread_x + thread_y;
        res.data[0] = src[r_idx];        
      } else if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::
                                          layout::col_major) {
        const int c_idx = thread_x + NumCols * thread_y;
        res.data[0] = src[c_idx];
      }
    }
  }
}

template <typename Group,
          sycl::ext::oneapi::experimental::matrix::layout Layout, typename T,
          size_t NumRows, size_t NumCols, size_t K, access::address_space Space,
          access::decorated IsDecorated>
void store_layoutT(
    joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, K, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride, Group &sg) {
  auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
              sg.get_local_linear_id();
  if constexpr (std::is_same_v<T, float>) {
    if constexpr (NumRows == 16 && NumCols == 16 && K == 4) {
      auto thread_x = idx % 16;
      auto thread_y = idx / 16;
      constexpr int batchStrideD = NumCols * NumRows;

      for (int b = 0; b < 4; ++b) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = thread_x + i * NumCols + thread_y * 4 * NumCols + b * batchStrideD;
          dst[d_idx] = src.wi_marray[i + b * 4];
        }
      }
    } else if constexpr (NumRows == 32 && NumCols == 32 && K == 8) {
      auto thread_x = idx % 32;
      auto thread_y = idx / 32;

      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx =
              thread_x + i * NumCols + thread_y * 4 * NumCols + j * 2 * 4 * NumCols;
          dst[d_idx] = src.wi_marray[i + 4 * j];
        }
      }
    } else if constexpr (NumRows == 32 && NumCols == 32 && K == 4) {
      auto thread_x = idx % 32;
      auto thread_y = idx / 32;

      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx =
              thread_x + i * NumCols + thread_y * 4 * NumCols + j * 2 * 4 * NumCols;
          dst[d_idx] = src.wi_marray[i + 4 * j];
        }
      }
    } else if constexpr (NumRows == 16 && NumCols == 16 && K == 16) {
      auto thread_x = idx % 16;
      auto thread_y = idx / 16;
      constexpr int batchStrideD = NumRows * NumCols;

      for (int i = 0; i < 4; ++i) {
        const int d_idx = thread_x + i * NumCols + thread_y * batchStrideD;
        dst[d_idx] = src.wi_marray[i];
      }
    } else if constexpr (NumRows == 4 && NumCols == 4 && K == 4) {
      auto thread_x = idx / 16;
      auto thread_y = idx % 16;
      constexpr int batchStrideD = NumRows * NumCols;

      for (int i = 0; i < 4; ++i) {
        const int d_idx = thread_x + i * NumCols + thread_y * 4 * NumCols;
        dst[d_idx] = src.wi_marray[i];
      }
    } else {
      static_assert(false && "Invalid dadimenstions!");
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if constexpr (NumRows == 16 && NumCols == 16 && K == 4) {
      auto thread_x = idx % 16;
      auto thread_y = idx / 16;
      constexpr int LDD = NumRows * NumCols;
    
      for (int i = 0; i < 4; ++i) {
        const int d_idx = thread_x + i * LDD + LDD * thread_y * 4;
        dst[d_idx] = src.wi_marray[i];
      }
    } else if constexpr (NumRows == 4 && NumCols == 4 && K == 4) {
      auto thread_x = idx / 4;
      auto thread_y = idx % 4;
      auto thread_xw = thread_x / 4;
      auto thread_xz = thread_x % 4;

      const int d_idx = thread_xw * K * NumRows + thread_xz + K * thread_y;
      dst[d_idx] = src.wi_marray[0];
    }
  }
}

template <typename Group, typename T, size_t NumRows, size_t NumCols, size_t K,
          access::address_space Space, access::decorated IsDecorated>
void joint_matrix_store_hip(
    joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, K, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
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
    typename Tm, typename Tc, std::size_t M, std::size_t N, std::size_t K,
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
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N, K,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &D,
    joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::a, M, N, K,
                      LayoutA> &A,
    joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::b, M, N, K,
                      LayoutB> &B,
    joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N, K,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &C) {
  if constexpr (std::is_same_v<Tm, sycl::half>) {
    if constexpr (M == 16 && N == 16 && K == 4) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_16x16x4f16(A.data, B.data,
                                                         C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 16 && N == 16 && K == 16) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_16x16x16f16(A.data, B.data,
                                                          C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 32 && N == 32 && K == 8) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_32x32x8f16(A.data, B.data,
                                                         C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 32 && N == 32 && K == 4) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_32x32x4f16(A.data, B.data,
                                                         C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 4 && N == 4 && K == 4) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_4x4x4f16(A.data, B.data,
                                                       C.wi_marray, 0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, bfloat16>) {
    if constexpr (M == 16 && N == 16 && K == 4) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(A.data, B.data,
                                                            C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 16 && N == 16 && K == 16) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A.data, B.data,
                                                             C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 32 && N == 32 && K == 8) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(A.data, B.data,
                                                             C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 32 && N == 32 && K == 4) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(A.data, B.data,
                                                             C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 4 && N == 4 && K == 4) {
      D.wi_marray = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(A.data, B.data,
                                                           C.wi_marray, 0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, double>) {
    if constexpr (M == 16 && N == 16 && K == 4) {
      D.wi_marray = __builtin_amdgcn_mfma_f64_16x16x4f64(A.data, B.data,
                                                         C.wi_marray, 0, 0, 0);
    } else if constexpr (M == 4 && N == 4 && K == 16) {
      D.wi_marray = __builtin_amdgcn_mfma_f64_4x4x4f64(A.data, B.data,
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
