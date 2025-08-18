cmake_minimum_required(VERSION 3.20)
project(spherical_averages VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable( test_sph_avg 
    test/test_sph_avg.cpp
)
target_include_directories( test_sph_avg PUBLIC include )

add_executable( test_for_beatrice
    test/test_for_beatrice.cpp
)
target_include_directories( test_for_beatrice PUBLIC include )

add_executable( test_beatrice_benchmark 
    test/test_beatrice_benchmark.cpp
)
target_include_directories( test_beatrice_benchmark PUBLIC include )
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    if(MSVC)
        target_compile_options( test_sph_avg PRIVATE /arch:AVX2) # For AVX2
        target_compile_options( test_sph_avg PRIVATE /openmp:experimental) # For OpenMP SIMD support
        target_compile_options( test_for_beatrice PRIVATE /arch:AVX2) # For AVX2
        target_compile_options( test_for_beatrice PRIVATE /openmp:experimental) # For OpenMP SIMD support
        target_compile_options( test_beatrice_benchmark PRIVATE /arch:AVX2) # For AVX2
        target_compile_options( test_beatrice_benchmark PRIVATE /openmp:experimental) # For OpenMP SIMD support
    else()
        set(CMAKE_CXX_COMPILER clang++)
        add_compile_options(-mavx2)
    endif()
endif()

