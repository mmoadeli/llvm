//===------- matrix-unified.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once
#include "matrix-intel.hpp"
#include "utils.hpp"
#include <sycl/ext/oneapi/matrix/matrix-tensorcores.hpp>
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#define __device__

#define __align__(x) alignas(x*16)

struct ThreadIndex {
  uint32_t x = 0;
  uint32_t y = 0;  
};

struct BlockIndex {
  uint32_t x = 0;
  uint32_t y = 0;  
};

struct BlockDim {
  uint32_t x = 0;
  uint32_t y = 0;  
};

const BlockIndex blockIdx;
const BlockDim blockDim;
const ThreadIndex threadIdx;

#include <sycl/ext/oneapi/matrix/rocwmma.hpp>

using matrix_use = sycl::ext::oneapi::experimental::matrix::use;
using matrix_layout = sycl::ext::oneapi::experimental::matrix::layout;

template <matrix_layout> struct to_hip_layout;

template <> struct to_hip_layout<matrix_layout::col_major> {
  using type = rocwmma::col_major;
};

template <> struct to_hip_layout<matrix_layout::row_major> {
  using type = rocwmma::row_major;
};

template <matrix_use, size_t, size_t> struct to_hip_use;

template <size_t Rows, size_t Cols> struct to_hip_use<matrix_use::accumulator, Rows, Cols> {
  using type = rocwmma::accumulator;
  static constexpr uint32_t BlockM = Rows;
  static constexpr uint32_t BlockN = Cols;
  static constexpr uint32_t BlockK = 0;
};

template <size_t Rows, size_t Cols> struct to_hip_use<matrix_use::a, Rows, Cols> {
  using type = rocwmma::matrix_a;
  static constexpr uint32_t BlockM = Rows;
  static constexpr uint32_t BlockN = 0;
  static constexpr uint32_t BlockK = Cols;
};

template <size_t Rows, size_t Cols> struct to_hip_use<matrix_use::b, Rows, Cols> {
  using type = rocwmma::matrix_b;
  static constexpr uint32_t BlockM = 0;
  static constexpr uint32_t BlockN = Rows;
  static constexpr uint32_t BlockK = Cols;
};

template <typename Type> struct to_hip_type;

template <> struct to_hip_type<sycl::ext::oneapi::bfloat16> {
  using type = rocwmma::bfloat16_t;
};

// template <> struct to_hip_type<sycl::float16_t> {
//   using type = float16_t;
// };

template <> struct to_hip_type<float> {
  using type = rocwmma::float32_t;
};

// template <> struct to_hip_type<float64_t> {
//   using type = rocwmma::float64_t;
// };

template <> struct to_hip_type<sycl::half> {
  using type = rocwmma::hfloat16_t;
};

template <> struct to_hip_type<int8_t> {
  using type = rocwmma::int8_t;
};

template <> struct to_hip_type<int32_t> {
  using type = rocwmma::int32_t;
};

template <> struct to_hip_type<uint8_t> {
  using type = rocwmma::uint8_t;
};

template <> struct to_hip_type<uint32_t> {
  using type = rocwmma::uint32_t;
};

#endif

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout>
struct joint_matrix {

#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__) && !defined(__HIP_PLATFORM_AMD__)
  sycl::ext::oneapi::detail::joint_matrix_cuda<T, Use, Rows, Cols, Layout>
      cuda_impl;
#elif defined(__HIP_PLATFORM_AMD__)
  using HipType = to_hip_type<T>::type;
  rocwmma::fragment<typename to_hip_use<Use, Rows, Cols>::type,
                    to_hip_use<Use, Rows, Cols>::BlockM,
                    to_hip_use<Use, Rows, Cols>::BlockN,
                    to_hip_use<Use, Rows, Cols>::BlockK, HipType,
                    typename to_hip_layout<Layout>::type> hip_impl;
#elif defined(__SPIR__)
  __spv::__spirv_JointMatrixINTEL<
      T, Rows, Cols, spv_matrix_layout_traits<Layout>::value,
      spv_scope_traits<Group>::value, spv_matrix_use_traits<Use>::value> *spvm;
#else
  static_assert(
      false,
      "The joint_matrix API is only supported by the Intel and CUDA backends");
#endif // defined(__NVPTX__)
#endif // defined(__SYCL_DEVICE_ONLY__)

  joint_matrix() {
#ifndef __SYCL_DEVICE_ONLY__
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__)
  // Generate a non-trivial assignment operator and copy c'tor that prevents
  // memcpy from being generated.
  // TODO: to remove, when either IGC can handle alloca JointMatrix or
  // combination of InstCombine + SROA + mem2reg can remove it
  joint_matrix(const joint_matrix &other) {
    spvm = other.spvm;
    return *this;
  }

  joint_matrix &operator=(const joint_matrix &rhs) {
    spvm = rhs.spvm;
    return *this;
  }
#endif // defined(__SPIR__)
#endif
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout>
class wi_data {

  joint_matrix<Group, T, Use, Rows, Cols, Layout> &jm;

  wi_data(joint_matrix<Group, T, Use, Rows, Cols, Layout> &_jm) : jm(_jm){};

  template <typename Grp, typename Type, use UseJm, size_t NumRows,
            size_t NumCols, layout LayoutJm>
  friend decltype(auto)
  get_wi_data(Grp,
              joint_matrix<Grp, Type, UseJm, NumRows, NumCols, LayoutJm> &);

public:
  size_t length() {
#if defined(__NVPTX__)
    return jm.cuda_impl.wi_marray.size();
#else
    throw runtime_error("get_wi_data is available using: "
                        "ext::intel::experimental::matrix::get_wi_data.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  };

  decltype(auto) operator[](size_t i) {
#if defined(__NVPTX__)
    return (jm.cuda_impl.wi_marray[i]);
#else
    throw runtime_error("get_wi_data is available using: "
                        "ext::intel::experimental::matrix::get_wi_data.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  };
};
#else
template <typename type, size_t size> class wi_data {
  marray<type, size> &data;
  wi_data(marray<type, size> &wi_marray) : data(wi_marray){};
  template <typename Grp, typename Type, use UseJm, size_t NumRows,
            size_t NumCols, layout LayoutJm>
  friend decltype(auto)
  get_wi_data(Grp,
              joint_matrix<Grp, Type, UseJm, NumRows, NumCols, LayoutJm> &);

public:
  size_t length() { return data.size(); };

  type &operator[](size_t i) { return data[i]; };
};
#endif

template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout>
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
__SYCL2020_DEPRECATED("get_wi_data() is deprecated for CUDA backend. Please "
                      "use joint_matrix_apply() instead.")
#else
__attribute__((unavailable(
    "get_wi_data can't be used on intel device, please use "
    "sycl::ext::intel::experimental::matrix::get_wi_data instead!")))
#endif
#endif
inline __SYCL_ALWAYS_INLINE decltype(auto)
    get_wi_data(Group sg, joint_matrix<Group, T, Use, Rows, Cols, Layout> &jm) {
#if defined(__SYCL_DEVICE_ONLY__)
  std::ignore = sg;
  return wi_data(jm);
#else
  std::ignore = sg;
  std::ignore = jm;
  if constexpr (std::is_same_v<T, precision::tf32>) {
    marray<float, 1> unused{};
    return wi_data<float, 1>(unused);
  } else {
    marray<T, 1> unused{};
    return wi_data<T, 1>(unused);
  }
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, use Use, size_t M, size_t N,
          layout Layout, typename F>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_apply(Group sg, joint_matrix<Group, T, Use, M, N, Layout> &jm,
                   F &&lambda) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  for (int i = 0; i < jm.cuda_impl.wi_marray.size(); i++) {
    lambda(jm.cuda_impl.wi_marray[i]);
  }
#else // NVPTX
  using storage_element_type =
      typename oneapi::detail::jm_type_interpretation_helper_trait<
          T>::storage_element_type;
  auto wi_data_c = sycl::ext::intel::experimental::matrix::get_wi_data(sg, jm);
  for (int i = 0; i < wi_data_c.length(); i++) {
    storage_element_type element = wi_data_c[i];
    lambda(element);
    wi_data_c[i] = element;
  }
#endif
#else
  std::ignore = sg;
  std::ignore = jm;
  std::ignore = lambda;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
  return;
}

template <typename Group, typename T, size_t NumRows, size_t NumCols, use Use,
          layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group sg,
                  joint_matrix<Group, T, Use, NumRows, NumCols, Layout> &res,
                  const T2 &v) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__) && !defined(__HIP_PLATFORM_AMD__)
  std::ignore = sg;
  res.cuda_impl.wi_marray = v;
#elif defined(__HIP_PLATFORM_AMD__)
  rocwmma::fill_fragment(res.hip_impl, v);
#else
  using storage_element_type =
      typename oneapi::detail::jm_type_interpretation_helper_trait<
          T>::storage_element_type;
  res.spvm =
      __spirv_CompositeConstruct<storage_element_type, T, NumRows, NumCols,
                                 spv_matrix_use_traits<Use>::value,
                                 spv_matrix_layout_traits<Layout>::value>(
          static_cast<storage_element_type>(v));
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = v;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value, bool> =
        true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg,
    joint_matrix<Group, S, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support load from private memory!");
#if defined(__NVPTX__) && !defined(__HIP_PLATFORM_AMD__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::load_accumulator_cuda(res.cuda_impl, src, stride,
                                                   Layout);
#elif defined(__HIP_PLATFORM_AMD__)
  using HipType = to_hip_type<S>::type;
  auto hip_layout = rocwmma::layout_t::mem_col_major; 
  if (Layout == layout::row_major)
    hip_layout = rocwmma::layout_t::mem_row_major;
  rocwmma::load_matrix_sync<typename to_hip_use<use::accumulator, NumRows, NumCols>::type,
                    to_hip_use<use::accumulator, NumRows, NumCols>::BlockM,
                    to_hip_use<use::accumulator, NumRows, NumCols>::BlockN,
                    to_hip_use<use::accumulator, NumRows, NumCols>::BlockK,
                    HipType>(res.hip_impl, src, stride, hip_layout);
#else
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(src);
  switch (Layout) {
  default:
    assert(false && "Invalid Memory Layout!");
  case layout::row_major:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        DecorT, S, NumRows, NumCols,
        spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::col_major:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        DecorT, S, NumRows, NumCols,
        spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case sycl::ext::intel::experimental::matrix::layout::packed:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        DecorT, S, NumRows, NumCols,
        spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::Packed,
        spv_scope_traits<Group>::value);
    break;
  }
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = src;
  std::ignore = stride;
  std::ignore = Layout;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, use Use, size_t NumRows,
    size_t NumCols, matrix::layout Layout, access::address_space Space,
    access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value ||
                         (std::is_same<S, precision::tf32>::value &&
                          std::is_same<std::remove_const_t<T>, float>::value),
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_load(Group sg,
                  joint_matrix<Group, S, Use, NumRows, NumCols, Layout> &res,
                  multi_ptr<T, Space, IsDecorated> src, size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support load from private memory!");
#if defined(__NVPTX__) && !defined(__HIP_PLATFORM_AMD__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::load_multiplicand_cuda<S, T, NumRows, NumCols, Use,
                                                    Layout, Space>(
      res.cuda_impl, src, stride);
#elif defined(__HIP_PLATFORM_AMD__)
  using HipType = to_hip_type<S>::type;
  rocwmma::load_matrix_sync<typename to_hip_use<Use, NumRows, NumCols>::type,
                    to_hip_use<Use, NumRows, NumCols>::BlockM,
                    to_hip_use<Use, NumRows, NumCols>::BlockN,
                    to_hip_use<Use, NumRows, NumCols>::BlockK,
                    HipType,
                    typename to_hip_layout<Layout>::type>(res.hip_impl, src, stride);
#else
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(src);
  res.spvm =
      __spirv_JointMatrixLoadINTEL<DecorT, S, NumRows, NumCols,
                                   spv_matrix_use_traits<Use>::value,
                                   spv_matrix_layout_traits<Layout>::value>(
          Ptr, stride, spv_matrix_layout_traits<Layout>::value,
          spv_scope_traits<Group>::value);
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = src;
  std::ignore = stride;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, typename S, size_t NumRows,
          size_t NumCols, access::address_space Space,
          access::decorated IsDecorated, layout LA>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group sg,
    joint_matrix<Group, S, use::accumulator, NumRows, NumCols, LA> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support store to private memory!");
#if defined(__NVPTX__) && !defined(__HIP_PLATFORM_AMD__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::joint_matrix_store_cuda<T, NumRows, NumCols,
                                                     Space>(src.cuda_impl, dst,
                                                            stride, Layout);
#elif defined(__HIP_PLATFORM_AMD__)
  using HipType = to_hip_type<S>::type;
  rocwmma::store_matrix_sync<typename to_hip_use<use::accumulator, NumRows, NumCols>::type,
                    to_hip_use<use::accumulator, NumRows, NumCols>::BlockM,
                    to_hip_use<use::accumulator, NumRows, NumCols>::BlockN,
                    to_hip_use<use::accumulator, NumRows, NumCols>::BlockK,
                    HipType,
                    typename to_hip_layout<LA>::type>(dst, src.hip_impl, stride);
#else
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(dst);
  switch (Layout) {
  default:
    assert(false && "Invalid Memory Layout!");
  case layout::row_major:
    __spirv_JointMatrixStoreINTEL<
        DecorT, T, NumRows, NumCols,
        spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::col_major:
    __spirv_JointMatrixStoreINTEL<
        DecorT, T, NumRows, NumCols,
        spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case sycl::ext::intel::experimental::matrix::layout::packed:
    __spirv_JointMatrixStoreINTEL<
        DecorT, T, NumRows, NumCols,
        spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::Packed,
        spv_scope_traits<Group>::value);
    break;
  }
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  std::ignore = Layout;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename Ta, typename Tb, typename Tc, std::size_t M,
          std::size_t K, std::size_t N, layout LayoutA, layout LayoutB, layout LayoutC>
inline __SYCL_ALWAYS_INLINE
    joint_matrix<Group, Tc, use::accumulator, M, N, LayoutC>
    joint_matrix_mad(
        Group sg, joint_matrix<Group, Ta, use::a, M, K, LayoutA> &A,
        joint_matrix<Group, Tb, use::b, K, N, LayoutB> &B,
        joint_matrix<Group, Tc, use::accumulator, M, N, LayoutC>
            &C) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__) && !defined(__HIP_PLATFORM_AMD__)
  std::ignore = sg;
  if constexpr (std::is_same<Ta, Tb>::value) {
    joint_matrix<Group, Tc, use::accumulator, M, N,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic>
        D;
    sycl::ext::oneapi::detail::joint_matrix_mad_cuda<Ta, Tc, M, K, N, LayoutA,
                                                     LayoutB>(
        D.cuda_impl, A.cuda_impl, B.cuda_impl, C.cuda_impl);
    return D;
  } else {
    assert(false && "Ta != Tb : In the CUDA backend joint_matrix_mad "
                    "requires that joint_matrix data types Ta and Tb match");
  }
#elif defined(__HIP_PLATFORM_AMD__)
  using HipTypeC = to_hip_type<Tc>::type;
  using HipTypeA = to_hip_type<Ta>::type;
  rocwmma::fragment<rocwmma::accumulator, M, N, K, HipTypeC,
                    typename to_hip_layout<LayoutC>::type> D;
  auto* AdjustedA = reinterpret_cast<
                      rocwmma::fragment<rocwmma::matrix_a, M, N, K, HipTypeA,
                                        typename to_hip_layout<LayoutA>::type>*>(&A.hip_impl);
  auto* AdjustedB = reinterpret_cast<
                      rocwmma::fragment<rocwmma::matrix_b, M, N, K, HipTypeA,
                                        typename to_hip_layout<LayoutB>::type>*>(&B.hip_impl);
  auto* AdjustedC = reinterpret_cast<
                      rocwmma::fragment<rocwmma::accumulator, M, N, K, HipTypeC,
                                        typename to_hip_layout<LayoutC>::type>*>(&C.hip_impl);
  rocwmma::mma_sync<M, N, K, HipTypeA, HipTypeC, 
                    typename to_hip_layout<LayoutA>::type,
                    typename to_hip_layout<LayoutB>::type,
                    typename to_hip_layout<LayoutC>::type,
                    typename to_hip_layout<LayoutC>::type>(
      D, *AdjustedA, *AdjustedB, *AdjustedC);
#else
  joint_matrix<Group, Tc, use::accumulator, M, N, layout::dynamic> res;
  if constexpr (std::is_same<Ta, uint16_t>::value &&
                std::is_same<Tb, uint16_t>::value &&
                std::is_same<Tc, float>::value)
    res.spvm = __spirv_JointMatrixMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_unsigned<Ta>::value && std::is_unsigned<Tb>::value)
    res.spvm = __spirv_JointMatrixUUMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_signed<Ta>::value && std::is_unsigned<Tb>::value)
    res.spvm = __spirv_JointMatrixSUMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_unsigned<Ta>::value && std::is_signed<Tb>::value)
    res.spvm = __spirv_JointMatrixUSMadINTEL(A.spvm, B.spvm, C.spvm);
  else
    res.spvm = __spirv_JointMatrixMadINTEL(A.spvm, B.spvm, C.spvm);
  return res;
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = A;
  std::ignore = B;
  std::ignore = C;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

// This function rounds the bottom 13 bits up or down, and then zeros out the
// bottom bits
inline __SYCL_ALWAYS_INLINE float round_to_tf32(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  int32_t tmp_int = __nvvm_f2tf32_rna(a);
  return __nvvm_bitcast_i2f(tmp_int);
#else
  return __spirv_RoundFToTF32INTEL(a);
#endif // defined(__NVPTX__)
#else
  uint32_t tmp_uint = reinterpret_cast<const uint32_t &>(a);
  tmp_uint += 0x1000u;
  tmp_uint &= 0xFFFFE000u;
  float ret = 0;
  std::memcpy(&ret, &tmp_uint, sizeof(float));
  return ret;
#endif // defined(__SYCL_DEVICE_ONLY__)
}
} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
