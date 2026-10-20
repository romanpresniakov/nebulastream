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

/// These tests cover: the fixed result type, value equality,
/// the reflection round-trip of a function without any fields, argument validation in the
/// factory, and the textual explain() form. The clock values themselves are tested in Clock.test.

#include <string>

#include <gtest/gtest.h>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <Functions/ArithmeticalFunctions/SubLogicalFunction.hpp>
#include <Functions/ClockLogicalFunction.hpp>
#include <Functions/ConstantValueLogicalFunction.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Schema/Field.hpp>
#include <Schema/Schema.hpp>
#include <Schema/SchemaFwd.hpp>
#include <Serialization/LogicalFunctionReflection.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <BaseUnitTest.hpp>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>

namespace NES
{
namespace
{
const auto expectedType = DataTypeProvider::provideDataType(DataType::Type::UINT64, DataType::NULLABLE::NOT_NULLABLE);

LogicalFunction constant(const DataType::Type type, const std::string& value)
{
    return ConstantValueLogicalFunction{DataTypeProvider::provideDataType(type), value};
}
}

TEST(ClockLogicalFunctionTest, ResultTypeIsNonNullableUint64)
{
    const auto function = ClockLogicalFunction{};
    EXPECT_EQ(function.getDataType(), expectedType);
    EXPECT_EQ(function.withInferredDataType(Schema<Field, Unordered>{}).getDataType(), expectedType);
}

TEST(ClockLogicalFunctionTest, SerializeDeserializeRoundtrip)
{
    const auto original = LogicalFunction{ClockLogicalFunction{}};
    const auto reflected = ReflectionContext{}.reflect(original);
    const auto deserialized = ReflectionContext{}.unreflect<LogicalFunction>(reflected);
    EXPECT_TRUE(original == deserialized);
    EXPECT_EQ(deserialized.getDataType(), expectedType);
}

TEST(ClockLogicalFunctionTest, SerializeDeserializeRoundtripAsChild)
{
    const auto original = LogicalFunction{SubLogicalFunction{ClockLogicalFunction{}, constant(DataType::Type::UINT64, "1000")}};
    const auto reflected = ReflectionContext{}.reflect(original);
    const auto deserialized = ReflectionContext{}.unreflect<LogicalFunction>(reflected);
    EXPECT_TRUE(original == deserialized);
}

TEST(ClockLogicalFunctionTest, Equality)
{
    EXPECT_TRUE(ClockLogicalFunction{} == ClockLogicalFunction{});
    EXPECT_FALSE(LogicalFunction{ClockLogicalFunction{}} == constant(DataType::Type::UINT64, "0"));
}

TEST(ClockLogicalFunctionTest, HasNoChildren)
{
    const auto function = ClockLogicalFunction{};
    EXPECT_TRUE(function.getChildren().empty());
    EXPECT_TRUE(function.withChildren({}) == function);
}

TEST(ClockLogicalFunctionTest, FactoryRejectsArguments)
{
    const auto created = ClockLogicalFunction::createClock(LogicalFunctionRegistryArguments{.children = {}});
    EXPECT_TRUE(created == LogicalFunction{ClockLogicalFunction{}});

    ASSERT_EXCEPTION_ERRORCODE(
        ClockLogicalFunction::createClock(LogicalFunctionRegistryArguments{.children = {constant(DataType::Type::UINT64, "1")}}),
        ErrorCode::CannotDeserialize);
}

TEST(ClockLogicalFunctionTest, ExplainUsesFunctionCallForm)
{
    EXPECT_EQ(ClockLogicalFunction{}.explain(ExplainVerbosity::Short), "Clock()");
}

}
