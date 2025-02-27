cmake_minimum_required(VERSION 3.12)
project(ChatGLM.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# third-party libraries
add_compile_definitions(GGML_CUDA_MMV_Y=2)  # for large vocab
include_directories(third_party/ggml/include/ggml third_party/ggml/src)
add_subdirectory(third_party/ggml)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "chatglm: disable sentencepiece shared libraries by default")
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "chatglm: disable tcmalloc by default")
include_directories(third_party/sentencepiece/src)
add_subdirectory(third_party/sentencepiece)

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
    set(CUDA_ARCHITECTURES "52;61;70;75;80;86" CACHE STRING "chatglm: cuda architectures to compile")
    set_property(TARGET ggml PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
endif ()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
    configure_file(third_party/ggml/src/ggml-metal.metal ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)
endif ()

if (GGML_PERF)
    add_compile_definitions(GGML_PERF)
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.cpp)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_FLAGS "-pedantic-errors")

add_library(chatglm STATIC chatglm.cpp)
target_link_libraries(chatglm PUBLIC ggml sentencepiece-static)

# c++ examples
option(CHATGLM_ENABLE_EXAMPLES "chatglm: enable c++ examples" ON)
if (CHATGLM_ENABLE_EXAMPLES)
    add_executable(main main.cpp)
    target_link_libraries(main PRIVATE chatglm)

    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        set(CHATGLM_OPENMP_TARGET OpenMP::OpenMP_CXX)
    endif ()
    add_executable(perplexity tests/perplexity.cpp)
    target_link_libraries(perplexity PRIVATE chatglm ${CHATGLM_OPENMP_TARGET})
endif ()

# GoogleTest
option(CHATGLM_ENABLE_TESTING "chatglm: enable testing" OFF)
if (CHATGLM_ENABLE_TESTING)
    enable_testing()

    # ref: https://github.com/google/googletest/blob/main/googletest/README.md
    include(FetchContent)
    FetchContent_Declare(
      googletest
      # Specify the commit you depend on and update it regularly.
      URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)

    # Now simply link against gtest or gtest_main as needed. Eg
    add_executable(chatglm_test chatglm_test.cpp)
    target_link_libraries(chatglm_test PRIVATE chatglm gtest_main)
    gtest_discover_tests(chatglm_test)
endif ()

option(CHATGLM_ENABLE_PYBIND "chatglm: enable python binding" OFF)
if (CHATGLM_ENABLE_PYBIND)
    set_target_properties(chatglm ggml sentencepiece-static PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    add_subdirectory(third_party/pybind11)
    pybind11_add_module(_C chatglm_pybind.cpp)
    target_link_libraries(_C PRIVATE chatglm)
endif ()

# lint
file(GLOB PY_SOURCES
    ${PROJECT_SOURCE_DIR}/chatglm_cpp/*.py
    ${PROJECT_SOURCE_DIR}/examples/*.py
    ${PROJECT_SOURCE_DIR}/tests/*.py
    ${PROJECT_SOURCE_DIR}/convert.py
    ${PROJECT_SOURCE_DIR}/setup.py)
add_custom_target(lint
    COMMAND clang-format -i ${CPP_SOURCES}
    COMMAND isort ${PY_SOURCES}
    COMMAND black ${PY_SOURCES} --verbose)

# mypy
add_custom_target(mypy
    mypy chatglm_cpp examples --exclude __init__.pyi
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# stub
add_custom_target(stub
    pybind11-stubgen chatglm_cpp -o .
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
endif ()
