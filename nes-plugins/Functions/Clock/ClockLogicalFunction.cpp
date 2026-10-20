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

#include <Functions/ClockLogicalFunction.hpp>

#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Schema/Field.hpp>
#include <Schema/Schema.hpp>
#include <Schema/SchemaFwd.hpp>
#include <Serialization/LogicalFunctionReflection.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>

namespace NES
{

/// The result type does not depend on any input, so it is fixed at construction.
ClockLogicalFunction::ClockLogicalFunction()
    : dataType(DataTypeProvider::provideDataType(DataType::Type::UINT64, DataType::NULLABLE::NOT_NULLABLE))
{
}

DataType ClockLogicalFunction::getDataType() const
{
    return dataType;
}

std::vector<LogicalFunction> ClockLogicalFunction::getChildren() const
{
    return {};
}

ClockLogicalFunction ClockLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.empty(), "ClockLogicalFunction takes no children, but got {}", children.size());
    return *this;
}

std::string_view ClockLogicalFunction::getType() const
{
    return NAME;
}

bool ClockLogicalFunction::operator==(const ClockLogicalFunction&) const
{
    return true;
}

std::string ClockLogicalFunction::explain(ExplainVerbosity) const
{
    return fmt::format("{}()", NAME);
}

LogicalFunction ClockLogicalFunction::withInferredDataType(const Schema<Field, Unordered>&) const
{
    return *this;
}

Reflected Reflector<ClockLogicalFunction>::operator()(const ClockLogicalFunction&, const ReflectionContext& context) const
{
    return context.reflect(detail::ReflectedClockLogicalFunction{});
}

ClockLogicalFunction Unreflector<ClockLogicalFunction>::operator()(const Reflected& reflected, const ReflectionContext& context) const
{
    std::ignore = context.unreflect<detail::ReflectedClockLogicalFunction>(reflected);
    return ClockLogicalFunction{};
}

LogicalFunctionRegistryReturnType ClockLogicalFunction::createClock(LogicalFunctionRegistryArguments arguments)
{
    if (not arguments.children.empty())
    {
        throw CannotDeserialize("Clock takes no arguments, but got {}", arguments.children.size());
    }
    return ClockLogicalFunction{};
}

}
