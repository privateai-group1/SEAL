# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(Bootcamp VERSION 3.4.5 LANGUAGES CXX)

# Executable will be in ../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Bootcamp_SOURCE_DIR}/../bin)


add_executable(bootcamp main.cpp)
target_sources(bootcamp
    PRIVATE
	example_ckks.cpp
	mvproduct.cpp
	polynomial_evaluation.cpp
	matrix_vector.cpp
	example_rnn.cpp
)

# Import Microsoft SEAL
find_package(SEAL 3.4.5 EXACT REQUIRED
    # Providing a path so this can be built without installing Microsoft SEAL
    PATHS ${Bootcamp_SOURCE_DIR}/../src/cmake
)

# Link Microsoft SEAL
target_link_libraries(bootcamp SEAL::seal)
