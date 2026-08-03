set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_mineru_llama_cpp binding.cpp)

install(TARGETS _mineru_llama_cpp DESTINATION mineru_llama_cpp)
