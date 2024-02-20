

# Conan automatically generated toolchain file
# DO NOT EDIT MANUALLY, it will be overwritten

# Avoid including toolchain file several times (bad if appending to variables like
#   CMAKE_CXX_FLAGS. See https://github.com/android/ndk/issues/323
include_guard()

message(STATUS "Using Conan toolchain: ${CMAKE_CURRENT_LIST_FILE}")

if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeToolchain' generator only works with CMake >= 3.15")
endif()




########## generic_system block #############
# Definition of system, platform and toolset
#############################################





set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64/cl.exe")




# Definition of VS runtime, defined from build_type, compiler.runtime, compiler.runtime_type
cmake_policy(GET CMP0091 POLICY_CMP0091)
if(NOT "${POLICY_CMP0091}" STREQUAL NEW)
    message(FATAL_ERROR "The CMake policy CMP0091 must be NEW, but is '${POLICY_CMP0091}'")
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>")

message(STATUS "Conan toolchain: C++ Standard 14 with extensions OFF")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Extra c, cxx, linkflags and defines


if(DEFINED CONAN_CXX_FLAGS)
  string(APPEND CMAKE_CXX_FLAGS_INIT " ${CONAN_CXX_FLAGS}")
endif()
if(DEFINED CONAN_C_FLAGS)
  string(APPEND CMAKE_C_FLAGS_INIT " ${CONAN_C_FLAGS}")
endif()
if(DEFINED CONAN_SHARED_LINKER_FLAGS)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " ${CONAN_SHARED_LINKER_FLAGS}")
endif()
if(DEFINED CONAN_EXE_LINKER_FLAGS)
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " ${CONAN_EXE_LINKER_FLAGS}")
endif()

get_property( _CMAKE_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE )
if(_CMAKE_IN_TRY_COMPILE)
    message(STATUS "Running toolchain IN_TRY_COMPILE")
    return()
endif()

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# Definition of CMAKE_MODULE_PATH
# the generators folder (where conan generates files, like this toolchain)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# Definition of CMAKE_PREFIX_PATH, CMAKE_XXXXX_PATH
# The Conan local "generators" folder, where this toolchain is saved.
list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR} )
list(PREPEND CMAKE_LIBRARY_PATH "C:/Users/fing.labcom/.conan2/p/b/sfmlead451e561c43/p/lib" "C:/Users/fing.labcom/.conan2/p/b/freetb02c70a110d95/p/lib" "C:/Users/fing.labcom/.conan2/p/b/libpndf00958a3b51b/p/lib" "C:/Users/fing.labcom/.conan2/p/b/zliba558161293c6a/p/lib" "C:/Users/fing.labcom/.conan2/p/b/bzip20b8af3a597035/p/lib" "C:/Users/fing.labcom/.conan2/p/b/brotl163e941a9c257/p/lib" "lib" "C:/Users/fing.labcom/.conan2/p/b/flacf885d7f201589/p/lib" "C:/Users/fing.labcom/.conan2/p/b/openad5184acb55d28/p/lib" "C:/Users/fing.labcom/.conan2/p/b/vorbi8887112ed7368/p/lib" "C:/Users/fing.labcom/.conan2/p/b/oggc3fcb9eeeda38/p/lib" "lib" "C:/Users/fing.labcom/.conan2/p/b/fmt270ea0f7cede4/p/lib")
list(PREPEND CMAKE_INCLUDE_PATH "C:/Users/fing.labcom/.conan2/p/b/sfmlead451e561c43/p/include" "C:/Users/fing.labcom/.conan2/p/b/freetb02c70a110d95/p/include" "C:/Users/fing.labcom/.conan2/p/b/freetb02c70a110d95/p/include/freetype2" "C:/Users/fing.labcom/.conan2/p/b/libpndf00958a3b51b/p/include" "C:/Users/fing.labcom/.conan2/p/b/zliba558161293c6a/p/include" "C:/Users/fing.labcom/.conan2/p/b/bzip20b8af3a597035/p/include" "C:/Users/fing.labcom/.conan2/p/b/brotl163e941a9c257/p/include" "C:/Users/fing.labcom/.conan2/p/b/brotl163e941a9c257/p/include/brotli" "include" "C:/Users/fing.labcom/.conan2/p/b/flacf885d7f201589/p/include" "C:/Users/fing.labcom/.conan2/p/b/openad5184acb55d28/p/include" "C:/Users/fing.labcom/.conan2/p/b/openad5184acb55d28/p/include/AL" "C:/Users/fing.labcom/.conan2/p/b/vorbi8887112ed7368/p/include" "C:/Users/fing.labcom/.conan2/p/b/oggc3fcb9eeeda38/p/include" "include" "C:/Users/fing.labcom/.conan2/p/b/fmt270ea0f7cede4/p/include")



if (DEFINED ENV{PKG_CONFIG_PATH})
set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR};$ENV{PKG_CONFIG_PATH}")
else()
set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR};")
endif()




# Variables
# Variables  per configuration


# Preprocessor definitions
# Preprocessor definitions per configuration


if(CMAKE_POLICY_DEFAULT_CMP0091)  # Avoid unused and not-initialized warnings
endif()