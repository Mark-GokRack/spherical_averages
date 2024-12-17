cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 20)


add_executable( test_sph_avg 
    include/SphAvg.hpp
    test/test_sph_avg.cpp
)

target_include_directories( test_sph_avg PUBLIC include )