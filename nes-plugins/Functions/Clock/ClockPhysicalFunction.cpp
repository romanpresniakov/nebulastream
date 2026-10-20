/*
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <Functions/ClockPhysicalFunction.hpp>

#include <chrono>
#include <cstdint>
#include <DataTypes/VarVal.hpp>
#include <Interface/Record.hpp>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>
#include <function.hpp>

namespace NES
{

VarVal ClockPhysicalFunction::execute(const Record&, ArenaRef&) const
{
    const auto nowMs = nautilus::invoke(
        +[]() -> uint64_t
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        });
    return VarVal{nowMs, false, false};
}

PhysicalFunctionRegistryReturnType ClockPhysicalFunction::createClock(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.empty(), "ClockPhysicalFunction takes no children, but got {}", arguments.childFunctions.size());
    return ClockPhysicalFunction{};
}

}
