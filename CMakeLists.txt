cmake_minimum_required(VERSION 3.20)
project(ml_inference_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Quality bar: warnings on everywhere, optional AddressSanitizer for leak checks.
add_compile_options(-Wall -Wextra)
# CI turns warnings into errors so a warning can never be merged unnoticed.
option(MIS_WERROR "Treat compiler warnings as errors" OFF)
if(MIS_WERROR)
  add_compile_options(-Werror)
endif()
option(ENABLE_ASAN "Build with AddressSanitizer" OFF)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# ---- Dependencies ----
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

# ONNX Runtime ships as a prebuilt tarball (see scripts/setup_deps.sh), not a CMake package,
# so we point at its folder and create an imported target by hand.
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime install folder")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

# ---- Core library: everything except main(), so tests can link it too ----
add_library(server_core STATIC
  src/version.cpp
  src/preprocess.cpp
  src/postprocess.cpp
  src/model_runner.cpp
  src/config.cpp
  src/metrics.cpp
  src/server.cpp
  src/batcher.cpp
  src/model_registry.cpp)
target_include_directories(server_core PUBLIC include)
# third_party headers are marked SYSTEM so their warnings don't break our -Wall build.
target_include_directories(server_core SYSTEM PUBLIC third_party)
target_link_libraries(server_core PUBLIC onnxruntime ${OpenCV_LIBS} Threads::Threads)
# httplib's default listen backlog is 5 pending connections; bursts beyond that are
# dropped by the kernel before our 503 logic ever runs. PUBLIC so every file that
# includes httplib.h sees the same value.
target_compile_definitions(server_core PUBLIC CPPHTTPLIB_LISTEN_BACKLOG=1024)
# Bake the ORT lib path into binaries so they run without setting LD_LIBRARY_PATH.
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")

add_executable(inference_server src/main.cpp)
target_link_libraries(inference_server PRIVATE server_core)

# CLI that classifies image files; used to compare C++ results against Python.
add_executable(classify tools/classify.cpp)
target_link_libraries(classify PRIVATE server_core)

# ---- Tests ----
# OFF in the Docker image build: no test download/compile needed to ship the server.
option(MIS_BUILD_TESTS "Build the GoogleTest unit tests" ON)
if(NOT MIS_BUILD_TESTS)
  return()
endif()
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(unit_tests
  tests/test_setup.cpp
  tests/test_preprocess.cpp
  tests/test_postprocess.cpp
  tests/test_model_runner.cpp
  tests/test_config.cpp
  tests/test_server.cpp
  tests/test_request_queue.cpp
  tests/test_batcher.cpp
  tests/test_model_registry.cpp
  tests/test_metrics.cpp)
target_link_libraries(unit_tests PRIVATE server_core GTest::gtest_main)
target_compile_definitions(unit_tests PRIVATE
  MIS_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
  MIS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
include(GoogleTest)
gtest_discover_tests(unit_tests)
