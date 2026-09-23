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

#include <ModelCatalog.hpp>

#include <cstddef>
#include <filesystem>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <BaseUnitTest.hpp>

#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/UnboundField.hpp>
#include <Identifiers/Identifier.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

namespace
{

DataType dt(DataType::Type type)
{
    return DataType{type, DataType::NULLABLE::NOT_NULLABLE};
}

/// Build a ModelFieldList with N auto-named fields of the given type.
ModelFieldList fields(size_t count, DataType::Type type)
{
    std::vector<UnqualifiedUnboundField> fieldVec;
    fieldVec.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        fieldVec.emplace_back(Identifier::parse(fmt::format("f{}", i)), dt(type));
    }
    return std::move(fieldVec) | std::ranges::to<ModelFieldList>();
}

ModelFieldList singleField(std::string_view name, DataType type)
{
    return std::vector{UnqualifiedUnboundField{Identifier::parse(std::string{name}), std::move(type)}} | std::ranges::to<ModelFieldList>();
}

/// Returns a copy of `schema` with the first field replaced by one of the same name but `replacementType`.
/// Schema is immutable, so we rebuild from scratch.
ModelFieldList replaceFirstFieldType(const ModelFieldList& schema, DataType replacementType)
{
    auto fields = schema | std::ranges::to<std::vector>();
    fields.front() = UnqualifiedUnboundField{fields.front().getFullyQualifiedName(), std::move(replacementType)};
    return std::move(fields) | std::ranges::to<ModelFieldList>();
}

ModelFieldList typedFields(const std::vector<DataType::Type>& types)
{
    std::vector<UnqualifiedUnboundField> fieldVec;
    fieldVec.reserve(types.size());
    for (size_t i = 0; i < types.size(); ++i)
    {
        fieldVec.emplace_back(Identifier::parse(fmt::format("f{}", i)), dt(types.at(i)));
    }
    return std::move(fieldVec) | std::ranges::to<ModelFieldList>();
}

/// Registration must fail with CannotLoadModel, and for the expected reason: several rules can
/// reject the same schema, so the message pins down which one fired.
void expectRegistrationRejected(ModelCatalog& catalog, const std::filesystem::path& path, ModelSchema schema, std::string_view expectedMessage)
{
    try
    {
        catalog.registerModel("m", path, std::move(schema));
        FAIL() << "Expected registration to be rejected with: " << expectedMessage;
    }
    catch (const Exception& ex)
    {
        EXPECT_EQ(ex.code(), NES::ErrorCode::CannotLoadModel);
        EXPECT_NE(std::string_view{ex.what()}.find(expectedMessage), std::string_view::npos) << ex.what();
    }
    EXPECT_FALSE(catalog.hasModel("m"));
}

std::filesystem::path identityPath()
{
    /// tiny_identity.onnx: f32, input shape [1,100], output shape [1,100] — 100 elements each side.
    return std::filesystem::path(INFERENCE_TEST_DATA) / "tiny_identity.onnx";
}

std::filesystem::path dynamicBatchPath()
{
    /// tiny_dynamic_batch.onnx: f32 identity with shape ["batch",100] on each side.
    return std::filesystem::path(INFERENCE_TEST_DATA) / "tiny_dynamic_batch.onnx";
}

std::filesystem::path fixedBatchPath()
{
    /// tiny_fixed_batch.onnx: f32 identity with shape [4,100] on each side.
    return std::filesystem::path(INFERENCE_TEST_DATA) / "tiny_fixed_batch.onnx";
}

std::filesystem::path twoInputsPath()
{
    /// tiny_two_inputs.onnx: f32 Concat(a [1,2], b [1,3]) -> y [1,5] — two input tensors, 5 output elements.
    return std::filesystem::path(INFERENCE_TEST_DATA) / "tiny_two_inputs.onnx";
}

}

class ModelCatalogTest : public ::testing::Test
{
};

/// NOLINTBEGIN(readability-magic-numbers)

TEST_F(ModelCatalogTest, RegistersModelWithMatchingFloat32Schema)
{
    ModelCatalog catalog;
    ASSERT_NO_THROW(catalog.registerModel(
        "identity",
        identityPath(),
        ModelSchema{.inputs = fields(100, DataType::Type::FLOAT32), .outputs = fields(100, DataType::Type::FLOAT32)}));
    EXPECT_TRUE(catalog.hasModel("identity"));
}

TEST_F(ModelCatalogTest, RegistersModelWithVarsizedSingleFieldOnBothSides)
{
    ModelCatalog catalog;
    ASSERT_NO_THROW(catalog.registerModel(
        "identity-varsized",
        identityPath(),
        ModelSchema{
            .inputs = singleField("blob_in", dt(DataType::Type::VARSIZED)),
            .outputs = singleField("blob_out", dt(DataType::Type::VARSIZED))}));
    EXPECT_TRUE(catalog.hasModel("identity-varsized"));
}

TEST_F(ModelCatalogTest, RejectsNonFloat32NonVarsizedInputType)
{
    ModelCatalog catalog;
    auto ins = replaceFirstFieldType(fields(100, DataType::Type::FLOAT32), dt(DataType::Type::INT32));
    ASSERT_EXCEPTION_ERRORCODE(
        catalog.registerModel("m", identityPath(), ModelSchema{.inputs = ins, .outputs = fields(100, DataType::Type::FLOAT32)}),
        NES::ErrorCode::CannotLoadModel);
}

TEST_F(ModelCatalogTest, RejectsNonFloat32NonVarsizedOutputType)
{
    ModelCatalog catalog;
    auto outs = replaceFirstFieldType(fields(100, DataType::Type::FLOAT32), dt(DataType::Type::INT64));
    ASSERT_EXCEPTION_ERRORCODE(
        catalog.registerModel("m", identityPath(), ModelSchema{.inputs = fields(100, DataType::Type::FLOAT32), .outputs = outs}),
        NES::ErrorCode::CannotLoadModel);
}

TEST_F(ModelCatalogTest, RejectsVarsizedMixedWithSiblings)
{
    ModelCatalog catalog;
    auto mixedInputs
        = std::
              vector{UnqualifiedUnboundField{Identifier::parse("blob"), dt(DataType::Type::VARSIZED)}, UnqualifiedUnboundField{Identifier::parse("tail"), dt(DataType::Type::FLOAT32)}}
        | std::ranges::to<ModelFieldList>();
    ASSERT_EXCEPTION_ERRORCODE(
        catalog.registerModel("m", identityPath(), ModelSchema{.inputs = mixedInputs, .outputs = fields(100, DataType::Type::FLOAT32)}),
        NES::ErrorCode::CannotLoadModel);
}

TEST_F(ModelCatalogTest, RejectsInputFieldCountMismatch)
{
    ModelCatalog catalog;
    ASSERT_EXCEPTION_ERRORCODE(
        catalog.registerModel(
            "m",
            identityPath(),
            /// model expects 100 elements
            ModelSchema{.inputs = fields(99, DataType::Type::FLOAT32), .outputs = fields(100, DataType::Type::FLOAT32)}),
        NES::ErrorCode::CannotLoadModel);
}

TEST_F(ModelCatalogTest, RejectsOutputFieldCountMismatch)
{
    ModelCatalog catalog;
    ASSERT_EXCEPTION_ERRORCODE(
        catalog.registerModel(
            "m",
            identityPath(),
            /// model produces 100 elements
            ModelSchema{.inputs = fields(100, DataType::Type::FLOAT32), .outputs = fields(10, DataType::Type::FLOAT32)}),
        NES::ErrorCode::CannotLoadModel);
}

/// A dynamic batch dimension resolves to 1, so the model registers against a schema
/// describing a single sample.
TEST_F(ModelCatalogTest, RegistersModelWithDynamicBatchDimension)
{
    ModelCatalog catalog;
    ASSERT_NO_THROW(catalog.registerModel(
        "m",
        dynamicBatchPath(),
        ModelSchema{.inputs = fields(100, DataType::Type::FLOAT32), .outputs = fields(100, DataType::Type::FLOAT32)}));
}

/// The declared schema is still validated against the resolved shape: a schema written
/// for a batch of 4 no longer matches once the dynamic dimension has become 1.
TEST_F(ModelCatalogTest, RejectsSchemaThatDoesNotMatchResolvedBatchDimension)
{
    ModelCatalog catalog;
    ASSERT_EXCEPTION_ERRORCODE(
        catalog.registerModel(
            "m",
            dynamicBatchPath(),
            ModelSchema{.inputs = fields(400, DataType::Type::FLOAT32), .outputs = fields(400, DataType::Type::FLOAT32)}),
        NES::ErrorCode::CannotLoadModel);
}

/// A model wanting several samples per invocation cannot be driven one tuple at a time.
TEST_F(ModelCatalogTest, RejectsModelWithFixedBatchDimension)
{
    ModelCatalog catalog;
    ASSERT_EXCEPTION_ERRORCODE(
        catalog.registerModel(
            "m",
            fixedBatchPath(),
            ModelSchema{.inputs = fields(400, DataType::Type::FLOAT32), .outputs = fields(400, DataType::Type::FLOAT32)}),
        NES::ErrorCode::CannotLoadModel);
}

/// One VARSIZED field per input tensor, mapped positionally.
TEST_F(ModelCatalogTest, RegistersModelWithOneVarsizedFieldPerInputTensor)
{
    ModelCatalog catalog;
    ASSERT_NO_THROW(catalog.registerModel(
        "two-inputs",
        twoInputsPath(),
        ModelSchema{.inputs = fields(2, DataType::Type::VARSIZED), .outputs = fields(5, DataType::Type::FLOAT32)}));
    EXPECT_TRUE(catalog.hasModel("two-inputs"));
}

TEST_F(ModelCatalogTest, RegistersModelWithMultipleVarsizedInputsAndVarsizedOutput)
{
    ModelCatalog catalog;
    ASSERT_NO_THROW(catalog.registerModel(
        "two-inputs-varsized",
        twoInputsPath(),
        ModelSchema{.inputs = fields(2, DataType::Type::VARSIZED), .outputs = singleField("blob_out", dt(DataType::Type::VARSIZED))}));
    EXPECT_TRUE(catalog.hasModel("two-inputs-varsized"));
}

TEST_F(ModelCatalogTest, RejectsFewerVarsizedInputsThanInputTensors)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        twoInputsPath(),
        ModelSchema{.inputs = fields(1, DataType::Type::VARSIZED), .outputs = fields(5, DataType::Type::FLOAT32)},
        "declared 1 varsized field(s) but has 2");
}

TEST_F(ModelCatalogTest, RejectsMoreVarsizedInputsThanInputTensors)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        twoInputsPath(),
        ModelSchema{.inputs = fields(3, DataType::Type::VARSIZED), .outputs = fields(5, DataType::Type::FLOAT32)},
        "declared 3 varsized field(s) but has 2");
}

TEST_F(ModelCatalogTest, RejectsMultipleVarsizedInputsOnSingleInputModel)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        identityPath(),
        ModelSchema{.inputs = fields(2, DataType::Type::VARSIZED), .outputs = fields(100, DataType::Type::FLOAT32)},
        "declared 2 varsized field(s) but has 1");
}

/// FLOAT32 fields map element-wise onto a single tensor; they cannot be spread across several.
TEST_F(ModelCatalogTest, RejectsFloat32InputsOnMultipleInputTensors)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        twoInputsPath(),
        ModelSchema{.inputs = fields(5, DataType::Type::FLOAT32), .outputs = fields(5, DataType::Type::FLOAT32)},
        "found multiple tensors");
}

/// Mixing is rejected regardless of which type comes first.
TEST_F(ModelCatalogTest, RejectsVarsizedFollowedByFloat32Input)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        twoInputsPath(),
        ModelSchema{
            .inputs = typedFields({DataType::Type::VARSIZED, DataType::Type::FLOAT32}), .outputs = fields(5, DataType::Type::FLOAT32)},
        "Mixing different types is not allowed");
}

TEST_F(ModelCatalogTest, RejectsFloat32FollowedByVarsizedInput)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        twoInputsPath(),
        ModelSchema{
            .inputs = typedFields({DataType::Type::FLOAT32, DataType::Type::VARSIZED}), .outputs = fields(5, DataType::Type::FLOAT32)},
        "Mixing different types is not allowed");
}

TEST_F(ModelCatalogTest, RejectsMixedOutputTypes)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        identityPath(),
        ModelSchema{
            .inputs = fields(100, DataType::Type::FLOAT32), .outputs = typedFields({DataType::Type::FLOAT32, DataType::Type::VARSIZED})},
        "Mixing different types is not allowed");
}

/// Relaxing the VARSIZED rule applies to inputs only; the output side keeps a single field.
TEST_F(ModelCatalogTest, RejectsMultipleVarsizedOutputs)
{
    ModelCatalog catalog;
    expectRegistrationRejected(
        catalog,
        twoInputsPath(),
        ModelSchema{.inputs = fields(2, DataType::Type::VARSIZED), .outputs = fields(2, DataType::Type::VARSIZED)},
        "VARSIZED requires exactly one Output field but got 2");
}

/// NOLINTEND(readability-magic-numbers)

}
