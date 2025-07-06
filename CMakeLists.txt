########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

cmake_minimum_required (VERSION 3.24)

include(FetchContent)

########################################################################
#
# Project: IgANet-PerfTests
#
########################################################################
project(IgANet-PerfTests)

add_library(iganet_perftests INTERFACE)
add_library(iganet::perftests ALIAS iganet_perftests)

########################################################################
# IgANets (core)
########################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  
  FetchContent_Declare(
    iganet_core
    GIT_REPOSITORY https://github.com/IgANets/iganet.git
    GIT_TAG modularized
  )
  set(IGANET_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)  
  FetchContent_MakeAvailable(iganet_core)
  FetchContent_GetProperties(iganet_core)
  include_directories(${iganet_core_SOURCE_DIR}/include)
  
  ######################################################################
  # Set C++ standard
  ######################################################################
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  enable_testing()
endif()

########################################################################
# GoogleTest
########################################################################
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

########################################################################
# BSplineLib
########################################################################
FetchContent_Declare(
  bsplinelib
  URL https://github.com/tataratat/BSplineLib/archive/refs/heads/main.zip
  )
set(BSPLINELIB_SHARED ON CACHE BOOL "" FORCE)
set(BSPLINELIB_BUILD_PUGIXML OFF CACHE BOOL "" FORCE)
set(pugixml_DIR ${pugixml_BINARY_DIR})
FetchContent_MakeAvailable(googletest bsplinelib)
FetchContent_GetProperties(bsplinelib)
include_directories(${bsplinelib_SOURCE_DIR}/Sources)

########################################################################
# Options
########################################################################
# Set a default coefficient numeric types if not specified
if(NOT IGANET_PERFTESTS_COEFF_TYPE)
  set (IGANET_PERFTESTS_COEFF_TYPE "double" CACHE STRING
    "Coefficient type(float, double)" FORCE)
endif()
set_property(CACHE IGANET_PERFTESTS_COEFF_TYPE PROPERTY STRINGS
  "float" "double")

message("")
message("PerfTests options:")
message("IGANET_PERFTESTS_COEFF_TYPE........: ${IGANET_PERFTESTS_COEFF_TYPE}")
message("")

########################################################################
# Config file
########################################################################
configure_file(${PROJECT_SOURCE_DIR}/src/perftest_config.hpp.in
  "${PROJECT_BINARY_DIR}/src/perftest_config.hpp" @ONLY)

include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

########################################################################
# Collect performance test sources
########################################################################
file(GLOB PERFTESTS src/*.cxx)

foreach(perftest ${PERFTESTS})

  get_filename_component(name ${perftest} NAME_WE)
  add_executable(${name} ${perftest})
  add_dependencies(iganet_perftests ${name})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_compile_options(${name} PRIVATE "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")
  target_link_libraries(${name} PRIVATE iganet::core)
  target_link_libraries(${name} PRIVATE splines)
  target_link_libraries(${name} PRIVATE gtest_main)
  target_compile_definitions(${name} PRIVATE UNITTEST_DISABLE_SHORT_MACROS)
  
  add_test(${name} ${name})
  
endforeach()
