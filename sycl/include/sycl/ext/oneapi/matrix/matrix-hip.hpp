
//===-------- matrix-hip.hpp - matrix ext impl ---*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once
#include "matrix-unified-utils.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>

#if defined(__gfx90a__)
#define __HIP_PLATFORM_AMD_MFMA__
#endif

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {} // namespace matrix
} // namespace experimental

using matrix_layout = sycl::ext::oneapi::experimental::matrix::layout;
using matrix_use = sycl::ext::oneapi::experimental::matrix::use;

namespace detail {

template <typename T, matrix_use Use, size_t Rows, size_t Cols,
          matrix_layout Layout = matrix_layout::dynamic, typename Cond = void>
struct joint_matrix_hip;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__HIP_PLATFORM_AMD_MFMA__)

template <typename T> struct to_hip_type {
  using type = T;
};

template <> struct to_hip_type<bfloat16> {
  using type = __bf16;
};

template <> struct to_hip_type<half> {
  using type = __fp16;
};

template <> struct to_hip_type<int8_t> {
  using type = int32_t;
};

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR(TYPE, USE, M, N, SIZE)                \
  template <matrix_layout Layout>                                              \
  struct joint_matrix_hip<                                                     \
      TYPE, matrix_use::USE, M, N, Layout,                                     \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    using ext_array_t = __attribute__((                                        \
        __vector_size__(SIZE * sizeof(typename to_hip_type<TYPE>::type))))     \
    typename to_hip_type<TYPE>::type;                                          \
    ext_array_t data = {0};                                                    \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 8, 32, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 8, 32, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 16, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 4, 16, 1)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(USE, M, N, SIZE)                 \
  template <matrix_layout Layout>                                              \
  struct joint_matrix_hip<                                                     \
      int8_t, matrix_use::USE, M, N, Layout,                                   \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    int8_t data[SIZE];                                                         \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(b, 8, 32, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(b, 16, 16, 4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N)                       \
  template <>                                                                  \
  struct joint_matrix_hip<TYPE, matrix_use::accumulator, M, N,                 \
                          matrix_layout::dynamic> {                            \
    using ext_array_t =                                                        \
        __attribute__((__vector_size__((M * N) / 64 * sizeof(TYPE)))) TYPE;    \
    ext_array_t data = {0};                                                    \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(double, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 16, 16)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC

template <matrix_layout Layout, typename S, typename T, size_t NumRows,
          size_t NumCols, access::address_space Space,
          access::decorated IsDecorated, typename Group>
void load_accumulator_layoutT(
    joint_matrix_hip<S, matrix_use::accumulator, NumRows, NumCols,
                     matrix_layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group &sg) {
  std::ignore = stride;
  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();
  if constexpr (Layout == matrix_layout::row_major) {
    res.data = {0};
  } else if constexpr (Layout == matrix_layout::col_major) {
    res.data = {0};
  }
};

template <typename Group, typename S, typename T, size_t NumRows,
          size_t NumCols, access::address_space Space,
          access::decorated IsDecorated>
void load_accumulator_hip(
    joint_matrix_hip<S, matrix_use::accumulator, NumRows, NumCols,
                     matrix_layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, matrix_layout Layout,
    Group &sg) {
  switch (Layout) {
  case matrix_layout::row_major:
    load_accumulator_layoutT<matrix_layout::row_major>(res, src, stride, sg);
    break;
  case matrix_layout::col_major:
    load_accumulator_layoutT<matrix_layout::col_major>(res, src, stride, sg);
    break;
  default:
    assert(false && "Invalid layout specified!");
  }
}

template <
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols,
    matrix_use Use, matrix_layout Layout, access::address_space Space,
    access::decorated IsDecorated,
    typename = typename std::enable_if_t<(Layout == matrix_layout::row_major ||
                                          Layout == matrix_layout::col_major)>>
void load_multiplicand_hip(
    joint_matrix_hip<S, Use, NumRows, NumCols, Layout> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group &sg) {
  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();
  std::ignore = stride;
  if constexpr (std::is_same_v<S, double>) {
    if constexpr (Layout == matrix_layout::row_major) {
      res.data[0] = src[idx];
    } else if constexpr (Layout == matrix_layout::col_major) {
      res.data[0] = src[(idx % NumRows) * 4 + idx / NumRows];
    }
  } else if constexpr (std::is_same_v<S, half> || std::is_same_v<S, bfloat16>) {
    if constexpr (NumRows == 16 && NumCols == 16) {
      const auto thread_x = idx % NumCols;
      const auto thread_y = idx / NumCols;
      constexpr int K = 16;

      if constexpr (Layout == matrix_layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx = thread_x * K + i + thread_y * 4;
          res.data[i] = src[c_idx];
        }
      } else if constexpr (Layout == matrix_layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x + i * NumCols + thread_y * NumCols * 4;
          res.data[i] = src[r_idx];
        }
      }
    } else if constexpr ((NumRows == 32 && NumCols == 8) ||
                         (NumRows == 8 && NumCols == 32)) {
      const auto thread_x = idx % 32;
      const auto thread_y = idx / 32;
      constexpr int K = 8;

      if constexpr (Layout == matrix_layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx = thread_x * K + i + thread_y * 4;
          res.data[i] = src[c_idx];
        }
      } else if constexpr (Layout == matrix_layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x + i * NumCols + thread_y * NumCols * 4;
          res.data[i] = src[r_idx];
        }
      }
    }
  } else if constexpr (std::is_same_v<S, int8_t>) {
    if constexpr (NumRows == 16 && NumCols == 16) {
      const auto thread_x = idx % NumCols;
      const auto thread_y = idx / NumCols;
      constexpr int K = 16;

      if constexpr (Layout == matrix_layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx = thread_x * K + i + thread_y * 4;
          res.data[i] = src[c_idx];
        }
      } else if constexpr (Layout == matrix_layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x + i * NumCols + thread_y * NumCols * 4;
          res.data[i] = src[r_idx];
        }
      }
    } else if constexpr ((NumRows == 32 && NumCols == 8) ||
                         (NumRows == 8 && NumCols == 32)) {
      const auto thread_x = idx % 32;
      const auto thread_y = idx / 32;
      constexpr int K = 8;

      if constexpr (Layout == matrix_layout::col_major) {
        for (int i = 0; i < 4; ++i) {
          const int c_idx = thread_x * K + i + thread_y * 4;
          res.data[i] = src[c_idx];
        }
      } else if constexpr (Layout == matrix_layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int r_idx = thread_x + i * NumCols + thread_y * NumCols * 4;
          res.data[i] = src[r_idx];
        }
      }
    }
  } else {
    static_assert(false && "Invalid layout specified6!");
  }
}

template <typename Group, matrix_layout Layout, typename T, size_t NumRows,
          size_t NumCols, access::address_space Space,
          access::decorated IsDecorated>
void store_layoutT(joint_matrix_hip<T, matrix_use::accumulator, NumRows,
                                    NumCols, matrix_layout::dynamic> &src,
                   multi_ptr<T, Space, IsDecorated> dst, size_t stride,
                   Group &sg) {
  std::ignore = stride;
  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();

  if constexpr (std::is_same_v<T, double>) {
    if constexpr (NumRows == 16 && NumCols == 16) {
      const auto thread_x = idx % NumCols;
      const auto thread_y = idx / NumCols;
      for (int i = 0; i < 4; ++i) {
        dst[thread_x + i * 4 * NumCols + thread_y * NumCols] = src.data[i];
      }
    }
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>) {
    if constexpr (NumRows == 16 && NumCols == 16) {
      const auto thread_x = idx % NumCols;
      const auto thread_y = idx / NumCols;
      constexpr int K = 16;

      for (int i = 0; i < 4; ++i) {
        const int d_idx = thread_x + i * K + thread_y * 4 * K;
        dst[d_idx] = src.data[i];
      }
    } else if constexpr (NumRows == 32 && NumCols == 32) {
      const auto thread_x = idx % NumCols;
      const auto thread_y = idx / NumCols;
      constexpr int K = 8;

      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx =
              thread_x + i * NumCols + thread_y * 4 * NumCols + j * 8 * NumCols;
          dst[d_idx] = src.data[i + 4 * j];
        }
      }
    }
  } else {
    static_assert(false && "Invalid dimenstions!");
  }
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
void joint_matrix_store_hip(
    joint_matrix_hip<T, matrix_use::accumulator, NumRows, NumCols,
                     matrix_layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride, matrix_layout Layout,
    Group &sg) {
  switch (Layout) {
  case matrix_layout::row_major:
    store_layoutT<Group, matrix_layout::row_major>(src, dst, stride, sg);
    break;
  case matrix_layout::col_major:
    store_layoutT<Group, matrix_layout::col_major>(src, dst, stride, sg);
    break;
  default:
    assert(false && "Invalid layout specified!");
  }
}

template <typename Tm, typename Tc, std::size_t M, std::size_t K, std::size_t N,
          matrix_layout LayoutA, matrix_layout LayoutB,
          std::enable_if_t<(LayoutA == matrix_layout::row_major ||
                            LayoutA == matrix_layout::col_major) &&
                               (LayoutB == matrix_layout::row_major ||
                                LayoutB == matrix_layout::col_major),
                           bool> = true>
void joint_matrix_mad_hip(joint_matrix_hip<Tc, matrix_use::accumulator, M, N,
                                           matrix_layout::dynamic> &D,
                          joint_matrix_hip<Tm, matrix_use::a, M, K, LayoutA> &A,
                          joint_matrix_hip<Tm, matrix_use::b, K, N, LayoutB> &B,
                          joint_matrix_hip<Tc, matrix_use::accumulator, M, N,
                                           matrix_layout::dynamic> &C) {
  if constexpr (std::is_same_v<Tm, sycl::half>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_f32_16x16x16f16(A.data, B.data, C.data, 0,
                                                     0, 0);
    } else if constexpr (M == 32 && N == 32) {
      D.data =
          __builtin_amdgcn_mfma_f32_32x32x8f16(A.data, B.data, C.data, 0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, bfloat16>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A.data, B.data, C.data,
                                                         0, 0, 0);
    } else if constexpr (M == 32 && N == 32 /* && K == 8 */) {
      D.data = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(A.data, B.data, C.data,
                                                        0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, double>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_f64_16x16x4f64(A.data[0], B.data[0],
                                                    C.data, 0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, int8_t>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_i32_16x16x16i8(
          *reinterpret_cast<int32_t *>(A.data),
          *reinterpret_cast<int32_t *>(B.data), C.data, 0, 0, 0);
    } else if constexpr (M == 32 && N == 32) {
      D.data = __builtin_amdgcn_mfma_i32_32x32x8i8(
          *reinterpret_cast<int32_t *>(A.data),
          *reinterpret_cast<int32_t *>(B.data), C.data, 0, 0, 0);
    }
  } else {
    static_assert(false && "Invalid configuration!");
  }
}

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl