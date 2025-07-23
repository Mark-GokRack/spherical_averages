cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_COMPILER clang++)

add_compile_options(-mavx2)

add_executable( test_sph_avg 
    test/test_sph_avg.cpp
)
target_include_directories( test_sph_avg PUBLIC include )

add_executable( test_beatrice_benchmark 
    test/test_beatrice_benchmark.cpp
)
target_include_directories( test_beatrice_benchmark PUBLIC include )