// make shared buffers for temporary variable.

#ifndef SSNSDP_BUFFER_H
#define SSNSDP_BUFFER_H

#include "core.h"
#include <array>
#include <bitset>
#include <type_traits>

namespace ssnsdp {

    constexpr Size BufferMaxCapacity = 5;

    static inline std::array<VectorX, BufferMaxCapacity> internal_buffers;
#ifdef SSNSDP_INTERNAL_DEBUG
    static inline std::bitset<BufferMaxCapacity> internal_buffers_in_use;
#endif

    // invalidate all buffers and set them to the same size
    inline void allocate_initialize(const Size size) {
        for (Index k = 0; k < BufferMaxCapacity; ++k) {
            auto& buf = internal_buffers[k];
            if (buf.size() < size) {
                buf = VectorX(size);
            }
#ifdef SSNSDP_INTERNAL_DEBUG
            internal_buffers_in_use[k] = false;
#endif
        }
    }

    // index: the unique identifier of each buffer.
    // Buffers of different indices are guaranteed not to interfere with each other.
    // The index-th buffer will be kept valid until it is freed,
    // or is reallocated by methods in this file.
    template <Index index = 0>
    inline Scalar* allocate_scalars(const Size size) {
        static_assert(index >= 0 && index < BufferMaxCapacity);
#ifdef SSNSDP_INTERNAL_DEBUG
        SSNSDP_ASSERT_MSG(!internal_buffers_in_use[index], "Requested buffer is in use");
        internal_buffers_in_use[index] = true;
#endif
        auto& buf = internal_buffers[index];
        if (buf.size() < size) {
            buf = VectorX(size);
        }
        return buf.data();
    }

    template <Index index = 0>
    SSNSDP_STRONG_INLINE
    Map<VectorX, Alignment> allocate_vector(const Size size) {
        Scalar* data = allocate_scalars<index>(size);
        return Map<VectorX, Alignment>(data, size);
    }

    template <Index index = 0>
    SSNSDP_STRONG_INLINE
    Map<MatrixX, Alignment> allocate_matrix(
        const Size rows,
        const Size cols) {
        Scalar* data = allocate_scalars<index>(rows * cols);
        return Map<MatrixX, Alignment>(data, rows, cols);
    }

    template <typename T>
    SSNSDP_STRONG_INLINE
    constexpr Size aligned_byte_size(const Size size) {
        return ((size * sizeof(T) - 1) / Alignment + 1) * Alignment;
    }

    template <typename T>
    SSNSDP_STRONG_INLINE
    constexpr T* aligned_offset(T* const ptr, const Size offset) {
        using CharPtr = std::conditional_t<std::is_const_v<T>, const char*, char*>;
        const auto byte_ptr = reinterpret_cast<CharPtr>(ptr);
        return reinterpret_cast<T*>(byte_ptr + aligned_byte_size<T>(offset));
    }

    template <typename Derived>
    SSNSDP_STRONG_INLINE
    Scalar* aligned_offset(const MatrixBase<Derived>& mat_) {
        Derived& mat = mat_.const_cast_derived();  // mat is not changed
        return aligned_offset(mat.data(), mat.rows() * mat.cols());
    }

    template <typename T>
    constexpr Size BufferMaxPadding = sizeof(T) > Alignment ? 1 :
        (Alignment - 1) / sizeof(T);

    // Only for debugging purpose.
    template <Index index = 0>
    SSNSDP_STRONG_INLINE
    void deactivate_buffer() {
        static_assert(index >= 0 && index < BufferMaxCapacity);
#ifdef SSNSDP_INTERNAL_DEBUG
        SSNSDP_ASSERT_MSG(internal_buffers_in_use[index], "Buffer not in use");
        internal_buffers_in_use[index] = false;
#endif
    }
}

#endif
