#pragma once
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

// カスタムアロケータ
template <typename T, std::size_t N>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, N>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }

        // メモリサイズを計算
        std::size_t size = n * sizeof(T);

        // Nバイト境界でアライメントされたメモリを確保
        // size が N の倍数である必要があることに注意が必要
        void* ptr = std::aligned_alloc( N, size );
        if( ptr == nullptr ){
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        std::free( ptr );
    }


    template<class U>
    struct rebind
    {
        //! 再束縛のためのアロケータ型
        using other = AlignedAllocator<U, N>;
    };
};

template <typename T, typename U, std::size_t N, std::size_t M>
bool operator==(const AlignedAllocator<T,N>&, const AlignedAllocator<U,M>&) noexcept {
    return ( N == M );
}

template <typename T, typename U, std::size_t N, std::size_t M>
bool operator!=(const AlignedAllocator<T,N>& lhs, const AlignedAllocator<U,M>& rhs) noexcept {
    return !(lhs==rhs);
}

// アライメントされたメモリを確保する vector クラス
template <typename T, std::size_t N>
using AlignedVector = std::vector<T, AlignedAllocator<T, N>>;