# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(${PROJECT_SOURCE_DIR}/cmake/RuntimeRegistrationUtil.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UnreflectionRegistrationUtil.cmake)

find_package(reflectcpp CONFIG REQUIRED)

add_library(clock_logical_function_plugin_library STATIC ClockLogicalFunction.cpp)
target_link_libraries(clock_logical_function_plugin_library PRIVATE nes-logical-operators reflectcpp::reflectcpp)
target_include_directories(clock_logical_function_plugin_library PUBLIC include PRIVATE .)
link_plugin_library(nes-logical-operators clock_logical_function_plugin_library)
add_registry_entry(LogicalFunction Clock)
add_unreflection_entry(LogicalFunction Clock)

add_library(clock_physical_function_plugin_library STATIC ClockPhysicalFunction.cpp)
target_link_libraries(clock_physical_function_plugin_library PRIVATE nes-physical-operators)
target_include_directories(clock_physical_function_plugin_library PUBLIC include PRIVATE .)
link_plugin_library(nes-physical-operators clock_physical_function_plugin_library)
add_registry_entry(PhysicalFunction Clock)

foreach (registry IN ITEMS LogicalFunction LogicalFunctionUnreflection PhysicalFunction)
    get_property(glue_lib GLOBAL PROPERTY "${registry}_RUNTIME_REGISTRY_GLUE_LIB")
    target_include_directories(${glue_lib} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/Functions)
endforeach ()

if (NES_ENABLES_TESTS)
    add_nes_unit_test(clock-logical-function-test ClockLogicalFunctionTest.cpp)
    target_link_libraries(clock-logical-function-test clock_logical_function_plugin_library nes-logical-operators nes-test-util reflectcpp::reflectcpp)
    target_include_directories(clock-logical-function-test PRIVATE include)
endif ()
