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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <Aggregation/Function/AggregationPhysicalFunction.hpp>
#include <DataTypes/DataType.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Interface/NautilusBuffer.hpp>
#include <Interface/Record.hpp>
#include <Interface/TimestampRef.hpp>
#include <Time/Timestamp.hpp>
#include <AggregationPhysicalFunctionRegistry.hpp>
#include <ExecutionContext.hpp>
#include <val_base.hpp>

namespace NES
{
/// Physical form of TENSOR_AGG (see TensorAggAggregationLogicalFunction). Lift appends one fixed-size entry per tuple
/// (its row, event timestamp, stream position and values) to pages from the buffer pool, combine attaches the other
/// state's pages, and lower allocates the tensor once per window and key and writes, for every cell, the value of the
/// latest entry by (timestamp, origin, sequence number, chunk number, position in the input buffer), or the default if
/// no entry wrote the cell. So the latest non-NULL value wins. At equal timestamps the tuple that came later in its
/// source's stream wins. A NULL value never overwrites a cell.
class TensorAggAggregationPhysicalFunction final : public AggregationPhysicalFunction
{
public:
    TensorAggAggregationPhysicalFunction(
        DataType resultType,
        Record::RecordFieldIdentifier resultFieldIdentifier,
        std::vector<PhysicalFunction> valueFunctions,
        std::vector<DataType> valueTypes,
        std::vector<PhysicalFunction> indexFunctions,
        std::vector<DataType> indexTypes,
        std::vector<uint64_t> indexDimensions,
        float defaultValue);

    void lift(
        const nautilus::val<AggregationState*>& aggregationState,
        BorrowedNautilusBuffer parentBuffer,
        PipelineMemoryProvider& pipelineMemoryProvider,
        const Record& record,
        const nautilus::val<Timestamp>& timestamp,
        const AggregationInputBuffer& inputBuffer) override;
    void combine(
        nautilus::val<AggregationState*> aggregationState1,
        BorrowedNautilusBuffer parentBuffer1,
        nautilus::val<AggregationState*> aggregationState2,
        BorrowedNautilusBuffer parentBuffer2,
        PipelineMemoryProvider& pipelineMemoryProvider) override;
    Record lower(
        nautilus::val<AggregationState*> aggregationState,
        BorrowedNautilusBuffer parentBuffer,
        PipelineMemoryProvider& pipelineMemoryProvider) override;
    void reset(
        nautilus::val<AggregationState*> aggregationState,
        BorrowedNautilusBuffer parentBuffer,
        PipelineMemoryProvider& pipelineMemoryProvider) override;
    void cleanup(nautilus::val<AggregationState*> aggregationState) override;
    [[nodiscard]] size_t getSizeOfStateInBytes() const override;
    static AggregationPhysicalFunctionRegistryReturnType create(AggregationPhysicalFunctionRegistryArguments arguments);

private:
    std::vector<PhysicalFunction> valueFunctions;
    std::vector<DataType> valueTypes;
    std::vector<PhysicalFunction> indexFunctions;
    std::vector<DataType> indexTypes;
    std::vector<uint64_t> indexDimensions;
    float defaultValue;
    uint64_t numberOfRows;
    bool hasNullFlags;
    uint64_t entrySize;
};
}
