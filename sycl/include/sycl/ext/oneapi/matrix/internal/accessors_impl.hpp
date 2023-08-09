/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_ACCESSORS_IMPL_HPP
#define ROCWMMA_ACCESSORS_IMPL_HPP

#include "accessors.hpp"
#include "io_config.hpp"
#include "io_shape.hpp"

// Fwd decl
namespace rocwmma
{
template <typename T, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t Rows, size_t Cols,
          sycl::ext::oneapi::experimental::matrix::layout Layout>
    class alignas(64) fragment;
}

namespace rocwmma
{
    ///
    /// DataType access
    ///

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetDataType<IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type = DataT;
    // };

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetDataType<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type = DataT;
    // };

    // ///
    // /// IOConfig access
    // ///

    template <typename DataT,
              sycl::ext::oneapi::experimental::matrix::use Use,
              size_t Rows,
              size_t Cols,
              sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    struct GetIOConfig<fragment<DataT, Use, Rows, Cols, DataLayout>>
    {
        using type =
            typename fragment<DataT, Use, Rows, Cols, DataLayout>::IOConfig;
    };

    // ///
    // /// IOShape access
    // ///

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetIOShape<IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type =
    //         typename IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::IOShape;
    // };

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetIOShape<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type = GetIOShape_t<
    //         GetIOConfig_t<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>>;
    // };

    // ///
    // /// IOTraits access
    // ///

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetIOTraits<IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type =
    //         typename IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::IOTraits;
    // };

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetIOTraits<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type = GetIOTraits_t<
    //         GetIOConfig_t<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>>;
    // };

    // ///
    // /// MatrixLayout access
    // ///

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetMatrixLayout<IOShape<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type =
    //         typename IOShape<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::MatrixLayout;
    // };

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetMatrixLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type = GetMatrixLayout_t<
    //         GetIOShape_t<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>>;
    // };

    // ///
    // /// MatrixLayout access
    // ///

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetDataLayout<IOShape<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type =
    //         typename IOShape<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::DataLayout;
    // };

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetDataLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type = GetDataLayout_t<
    //         GetIOShape_t<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>>;
    // };

} // namespace rocwmma

#endif // ROCWMMA_ACCESSORS_IMPL_HPP
