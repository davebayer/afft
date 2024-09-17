########################################################################################################################
#
# This file is part of afft library.
#
# Copyright (c) 2024 David Bayer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
########################################################################################################################

function(add_examples mpBackend target)
  string(TOLOWER "${mpBackend}" mpBackend)
  string(TOLOWER "${target}" target)

  set(EXAMPLES_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/${mpBackend}/${target}")
  file(GLOB EXAMPLES "${EXAMPLES_SRC_DIR}/*.cpp" "${EXAMPLES_SRC_DIR}/*.c")
  
  foreach(EXAMPLE ${EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
    get_filename_component(EXAMPLE_EXT  ${EXAMPLE} LAST_EXT)
    string(SUBSTRING ${EXAMPLE_EXT} 1 -1 EXAMPLE_EXT)
    string(FIND ${EXAMPLE_NAME} "_mod" USES_MODULE)

    if(NOT (USES_MODULE EQUAL -1) AND NOT TARGET afft-module)
      continue()
    endif()

    set(EXAMPLE_TARGET example-${mpBackend}-${target}-${EXAMPLE_NAME}-${EXAMPLE_EXT})
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE})
    set_target_properties(${EXAMPLE_TARGET}
      PROPERTIES
        OUTPUT_NAME              "${EXAMPLE_NAME}-${EXAMPLE_EXT}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${mpBackend}/${target}")
    target_include_directories(${EXAMPLE_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    
    
    if(EXAMPLE_EXT STREQUAL ".c")
      target_compile_features(${EXAMPLE_TARGET} PRIVATE c_std_99)
      target_link_libraries(${EXAMPLE_TARGET} PRIVATE afft::afft)
    else()
      if(NOT (USES_MODULE EQUAL -1))
        target_compile_features(${EXAMPLE_TARGET} PRIVATE cxx_std_20)
        set_target_properties(${EXAMPLE_TARGET} PROPERTIES CXX_SCAN_FOR_MODULES TRUE)
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE afft::afft-module)
      else()
        target_compile_features(${EXAMPLE_TARGET} PRIVATE cxx_std_17)
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE afft::afft)
      endif()
    endif()
  endforeach()
endfunction()

foreach(target ${AFFT_TARGETS})
  add_examples("" ${target})
endforeach()

foreach(mpBackend IN LISTS ${AFFT_MP_BACKEND})
  foreach(target ${TARGET_LIST})
    add_examples(${mpBackend} ${target})
  endforeach()
endforeach()
