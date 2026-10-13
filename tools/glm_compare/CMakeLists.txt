# tools/glm_compare: llama_logits (llama_logits.cpp) built against a llama.cpp checkout that has glm5-next (upstream,
# not third_party/llama.cpp, which predates it).  CPU only: the reference.
#
#   cmake -S tools/glm_compare -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR=<checkout>
#   cmake --build <build> --target llama_logits
cmake_minimum_required(VERSION 3.14)
project(glm_compare C CXX)
set(LLAMA_CPP_DIR "" CACHE PATH "a llama.cpp checkout with glm5-next")
if(NOT EXISTS "${LLAMA_CPP_DIR}/src/models/glm5-next.cpp")
  message(FATAL_ERROR "LLAMA_CPP_DIR (${LLAMA_CPP_DIR}) is not a llama.cpp checkout with glm5-next")
endif()
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)
add_executable(llama_logits llama_logits.cpp)
target_link_libraries(llama_logits PRIVATE llama)
