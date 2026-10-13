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

#include <Aggregation/Function/TensorAggAggregationPhysicalFunction.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>
#include <Aggregation/Function/AggregationPhysicalFunction.hpp>
#include <DataTypes/DataType.hpp>
#include <DataTypes/VarVal.hpp>
#include <DataTypes/VariableSizedData.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Interface/NautilusBuffer.hpp>
#include <Interface/Record.hpp>
#include <Interface/TimestampRef.hpp>
#include <Operators/Windows/Aggregations/TensorAggAggregationLogicalFunction.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Time/Timestamp.hpp>
#include <nautilus/function.hpp>
#include <AggregationPhysicalFunctionRegistry.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <val_arith.hpp>
#include <val_bool.hpp>
#include <val_ptr.hpp>

namespace NES
{
namespace
{

struct TensorAggState
{
    ChildBufferIndex::Underlying pagesIndex = 0;
    int8_t* currentPage = nullptr;
    uint64_t writeOffset = 0;
    uint64_t pageSize = 0;
    uint64_t originId = 0;
    uint64_t sequenceNumber = 0;
    uint64_t chunkNumber = 0;
    uint64_t positionInBuffer = 0;
    bool hasInputBuffer = false;
};

struct PageHeader
{
    uint64_t numberOfEntries;
};

struct EntryHeader
{
    uint64_t row;
    uint64_t timestamp;
    uint64_t originId;
    uint64_t sequenceNumber;
    uint64_t chunkNumber;
    uint64_t positionInBuffer;
};

constexpr uint64_t ENTRY_ALIGNMENT = 8;

void createPageList(TensorAggState* state, TupleBuffer* parentBuffer, AbstractBufferProvider* bufferProvider)
{
    PRECONDITION(state != nullptr && parentBuffer != nullptr && bufferProvider != nullptr, "TENSOR_AGG reset needs its state and buffers");
    auto root = bufferProvider->getUnpooledBuffer(1);
    if (not root)
    {
        throw BufferAllocationFailure("No unpooled TupleBuffer available for the TENSOR_AGG page list");
    }
    const auto rootIndex = parentBuffer->storeChildBuffer(*root).getRawValue();
    new (state) TensorAggState{.pagesIndex = rootIndex}; /// NOLINT(cppcoreguidelines-owning-memory)
}


int8_t* appendEntry(
    TensorAggState* state,
    TupleBuffer* parentBuffer,
    AbstractBufferProvider* bufferProvider,
    const uint64_t entrySize,
    const uint64_t row,
    const uint64_t timestamp,
    const OriginId originId,
    const SequenceNumber sequenceNumber,
    const ChunkNumber chunkNumber)
{
    PRECONDITION(state != nullptr && parentBuffer != nullptr && bufferProvider != nullptr, "TENSOR_AGG lift needs its state and buffers");
    if (not state->hasInputBuffer || state->originId != originId.getRawValue() || state->sequenceNumber != sequenceNumber.getRawValue()
        || state->chunkNumber != chunkNumber.getRawValue())
    {
        state->originId = originId.getRawValue();
        state->sequenceNumber = sequenceNumber.getRawValue();
        state->chunkNumber = chunkNumber.getRawValue();
        state->positionInBuffer = 0;
        state->hasInputBuffer = true;
    }

    if (state->currentPage == nullptr || state->writeOffset + entrySize > state->pageSize)
    {
        auto page = bufferProvider->getBufferBlocking();
        if (page.getBufferSize() < sizeof(PageHeader) + entrySize)
        {
            throw UnsupportedQuery(
                "A TENSOR_AGG entry of {} bytes does not fit into a buffer of {} bytes; use fewer value fields per position",
                entrySize,
                page.getBufferSize());
        }
        auto* const pageData = reinterpret_cast<int8_t*>(page.getAvailableMemoryArea<>().data()); /// NOLINT
        new (pageData) PageHeader{.numberOfEntries = 0}; /// NOLINT(cppcoreguidelines-owning-memory)
        state->currentPage = pageData;
        state->writeOffset = sizeof(PageHeader);
        state->pageSize = page.getBufferSize();
        auto pages = parentBuffer->loadChildBuffer(ChildBufferIndex{state->pagesIndex});
        std::ignore = pages.storeChildBuffer(page);
    }

    auto* const entry = state->currentPage + state->writeOffset; /// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    state->writeOffset += entrySize;
    reinterpret_cast<PageHeader*>(state->currentPage)->numberOfEntries += 1; /// NOLINT
    std::memset(entry, 0, entrySize);
    new (entry) EntryHeader{ /// NOLINT(cppcoreguidelines-owning-memory)
        .row = row,
        .timestamp = timestamp,
        .originId = state->originId,
        .sequenceNumber = state->sequenceNumber,
        .chunkNumber = state->chunkNumber,
        .positionInBuffer = state->positionInBuffer++};
    return entry + sizeof(EntryHeader); /// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

void attachPages(TensorAggState* destination, TupleBuffer* destinationParent, const TensorAggState* source, const TupleBuffer* sourceParent)
{
    PRECONDITION(destination != nullptr && source != nullptr, "TENSOR_AGG states must not be null");
    auto destinationPages = destinationParent->loadChildBuffer(ChildBufferIndex{destination->pagesIndex});
    const auto sourcePages = sourceParent->loadChildBuffer(ChildBufferIndex{source->pagesIndex});
    for (uint32_t page = 0; page < sourcePages.getNumberOfChildBuffers(); ++page)
    {
        auto sharedPage = sourcePages.loadChildBuffer(ChildBufferIndex{page});
        std::ignore = destinationPages.storeChildBuffer(sharedPage);
    }
}

void materializeTensor(
    const TensorAggState* state,
    const TupleBuffer* parentBuffer,
    int8_t* tensor,
    const uint64_t numberOfRows,
    const uint64_t rowWidth,
    const float defaultValue,
    const uint64_t entrySize,
    const uint64_t hasNullFlags)
{
    PRECONDITION(state != nullptr && parentBuffer != nullptr && tensor != nullptr, "TENSOR_AGG lower needs its state and buffers");
    for (uint64_t i = 0; i < numberOfRows * rowWidth; ++i)
    {
        std::memcpy(tensor + (i * sizeof(float)), &defaultValue, sizeof(float)); /// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) // fill in default values
    }

    std::vector<const EntryHeader*> entries;
    const auto pages = parentBuffer->loadChildBuffer(ChildBufferIndex{state->pagesIndex});
    for (uint32_t i = 0; i < pages.getNumberOfChildBuffers(); ++i)
    {
        const auto page = pages.loadChildBuffer(ChildBufferIndex{i});
        const auto* const pageData = reinterpret_cast<const int8_t*>(page.getAvailableMemoryArea<>().data()); /// NOLINT
        const auto numberOfEntries = reinterpret_cast<const PageHeader*>(pageData)->numberOfEntries; /// NOLINT
        for (uint64_t j = 0; j < numberOfEntries; ++j)
        {
            entries.push_back(reinterpret_cast<const EntryHeader*>(pageData + sizeof(PageHeader) + j * entrySize)); /// NOLINT
        }
    }
    std::ranges::sort(
        entries,
        std::less<>{},
        [](const EntryHeader* entry)
        { return std::tie(entry->timestamp, entry->originId, entry->sequenceNumber, entry->chunkNumber, entry->positionInBuffer); });

    for (const auto* entry : entries)
    {
        const auto* const values = reinterpret_cast<const int8_t*>(entry) + sizeof(EntryHeader); /// NOLINT
        const auto* const nullFlags = values + (rowWidth * sizeof(float)); /// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        auto* const row = tensor + (entry->row * rowWidth * sizeof(float)); /// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        for (uint64_t i = 0; i < rowWidth; ++i)
        {
            if (hasNullFlags == 0 || nullFlags[i] == 0)
            {
                std::memcpy(row + i * sizeof(float), values + i * sizeof(float), sizeof(float)); /// NOLINT
            }
        }
    }
}

uint64_t entrySizeFor(const uint64_t rowWidth, const bool hasNullFlags)
{
    const auto unpadded = sizeof(EntryHeader) + rowWidth * sizeof(float) + (hasNullFlags ? rowWidth : 0);
    return (unpadded + ENTRY_ALIGNMENT - 1) / ENTRY_ALIGNMENT * ENTRY_ALIGNMENT;
}
}

TensorAggAggregationPhysicalFunction::TensorAggAggregationPhysicalFunction(
    DataType resultType,
    Record::RecordFieldIdentifier resultFieldIdentifier,
    std::vector<PhysicalFunction> valueFunctions,
    std::vector<DataType> valueTypes,
    std::vector<PhysicalFunction> indexFunctions,
    std::vector<DataType> indexTypes,
    std::vector<uint64_t> indexDimensions,
    const float defaultValue)
    : AggregationPhysicalFunction(valueTypes.front(), std::move(resultType), valueFunctions.front(), std::move(resultFieldIdentifier))
    , valueFunctions(std::move(valueFunctions))
    , valueTypes(std::move(valueTypes))
    , indexFunctions(std::move(indexFunctions))
    , indexTypes(std::move(indexTypes))
    , indexDimensions(std::move(indexDimensions))
    , defaultValue(defaultValue)
    , numberOfRows(std::accumulate(this->indexDimensions.begin(), this->indexDimensions.end(), uint64_t{1}, std::multiplies<>()))
    , hasNullFlags(std::ranges::any_of(this->valueTypes, [](const DataType& type) { return type.nullable; }))
    , entrySize(entrySizeFor(this->valueFunctions.size(), hasNullFlags))
{
    PRECONDITION(this->valueFunctions.size() == this->valueTypes.size(), "TENSOR_AGG needs a type for every value function");
    PRECONDITION(
        this->indexFunctions.size() == this->indexTypes.size() && this->indexFunctions.size() == this->indexDimensions.size(),
        "TENSOR_AGG needs a type and a dimension for every index function");
}

void TensorAggAggregationPhysicalFunction::lift(
    const nautilus::val<AggregationState*>& aggregationState,
    BorrowedNautilusBuffer parentBuffer,
    PipelineMemoryProvider& pipelineMemoryProvider,
    const Record& record,
    const nautilus::val<Timestamp>& timestamp,
    const AggregationInputBuffer& inputBuffer)
{
    nautilus::val<uint64_t> row{0};
    nautilus::val<bool> insideTensor{true};
    for (nautilus::static_val<size_t> i = 0; i < indexFunctions.size(); ++i)
    {
        const auto index = indexFunctions[i].execute(record, pipelineMemoryProvider.arena);
        const auto signedPosition = index.getRawValueAs<nautilus::val<int64_t>>();
        insideTensor = insideTensor && signedPosition >= nautilus::val<int64_t>{0}
            && signedPosition < nautilus::val<int64_t>{static_cast<int64_t>(indexDimensions[i])};
        if (indexTypes[i].nullable)
        {
            insideTensor = insideTensor && not index.isNull();
        }
        row = row * nautilus::val<uint64_t>{indexDimensions[i]} + index.getRawValueAs<nautilus::val<uint64_t>>();
    }

    if (insideTensor)
    {
        const auto values = nautilus::invoke(
            appendEntry,
            static_cast<nautilus::val<TensorAggState*>>(aggregationState),
            parentBuffer.asArg(),
            pipelineMemoryProvider.bufferProvider,
            nautilus::val<uint64_t>{entrySize},
            row,
            timestamp.convertToValue(),
            inputBuffer.originId,
            inputBuffer.sequenceNumber,
            inputBuffer.chunkNumber);
        for (nautilus::static_val<size_t> i = 0; i < valueFunctions.size(); ++i)
        {
            const auto value = valueFunctions[i].execute(record, pipelineMemoryProvider.arena);
            const auto target = values + nautilus::val<uint64_t>{i * sizeof(float)};
            if (valueTypes[i].nullable)
            {
                if (value.isNull())
                {
                    const auto nullFlag = values + nautilus::val<uint64_t>{valueFunctions.size() * sizeof(float) + i};
                    VarVal{nautilus::val<int8_t>{1}}.writeToMemory(nullFlag);
                }
                else
                {
                    VarVal{value.getRawValueAs<nautilus::val<float>>()}.writeToMemory(target);
                }
            }
            else
            {
                VarVal{value.getRawValueAs<nautilus::val<float>>()}.writeToMemory(target);
            }
        }
    }
}

void TensorAggAggregationPhysicalFunction::combine(
    nautilus::val<AggregationState*> aggregationState1,
    BorrowedNautilusBuffer parentBuffer1,
    nautilus::val<AggregationState*> aggregationState2,
    BorrowedNautilusBuffer parentBuffer2,
    PipelineMemoryProvider&)
{
    nautilus::invoke(
        attachPages,
        static_cast<nautilus::val<TensorAggState*>>(aggregationState1),
        parentBuffer1.asArg(),
        static_cast<nautilus::val<TensorAggState*>>(aggregationState2),
        parentBuffer2.asArg());
}

Record TensorAggAggregationPhysicalFunction::lower(
    const nautilus::val<AggregationState*> aggregationState,
    BorrowedNautilusBuffer parentBuffer,
    PipelineMemoryProvider& pipelineMemoryProvider)
{
    const auto tensorBytes = nautilus::val<uint64_t>{numberOfRows * valueFunctions.size() * sizeof(float)};
    auto payload = pipelineMemoryProvider.arena.allocateVariableSizedData(tensorBytes);
    nautilus::invoke(
        materializeTensor,
        static_cast<nautilus::val<TensorAggState*>>(aggregationState),
        parentBuffer.asArg(),
        payload.getContent(),
        nautilus::val<uint64_t>{numberOfRows},
        nautilus::val<uint64_t>{valueFunctions.size()},
        nautilus::val<float>{defaultValue},
        nautilus::val<uint64_t>{entrySize},
        nautilus::val<uint64_t>{hasNullFlags ? uint64_t{1} : uint64_t{0}});
    Record result;
    result.write(resultFieldIdentifier, VarVal{payload});
    return result;
}

void TensorAggAggregationPhysicalFunction::reset(
    const nautilus::val<AggregationState*> aggregationState,
    BorrowedNautilusBuffer parentBuffer,
    PipelineMemoryProvider& pipelineMemoryProvider)
{
    nautilus::invoke(
        createPageList,
        static_cast<nautilus::val<TensorAggState*>>(aggregationState),
        parentBuffer.asArg(),
        pipelineMemoryProvider.bufferProvider);
}

void TensorAggAggregationPhysicalFunction::cleanup(nautilus::val<AggregationState*>)
{
    /// The pages are children of the parent hash-map buffer and are released with it.
}

size_t TensorAggAggregationPhysicalFunction::getSizeOfStateInBytes() const
{
    return sizeof(TensorAggState);
}

AggregationPhysicalFunctionRegistryReturnType
TensorAggAggregationPhysicalFunction::create(AggregationPhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.logicalFunction.has_value(), "TENSOR_AGG requires its logical function to be lowered");
    const auto logical = arguments.logicalFunction->getAs<TensorAggAggregationLogicalFunction>();
    const auto numberOfValues = logical->getValueFunctions().size();
    const auto numberOfIndices = logical->getIndexFunctions().size();
    PRECONDITION(
        arguments.additionalInputFunctions.size() == numberOfValues - 1 + numberOfIndices
            && arguments.additionalInputTypes.size() == arguments.additionalInputFunctions.size(),
        "TENSOR_AGG expects its value and index fields as inputs");

    std::vector<PhysicalFunction> valueFunctions{std::move(arguments.inputFunction)};
    std::vector<DataType> valueTypes{std::move(arguments.inputType)};
    std::vector<PhysicalFunction> indexFunctions;
    std::vector<DataType> indexTypes;
    for (size_t input = 0; input < arguments.additionalInputFunctions.size(); ++input)
    {
        const bool isValue = input < numberOfValues - 1;
        (isValue ? valueFunctions : indexFunctions).push_back(std::move(arguments.additionalInputFunctions[input]));
        (isValue ? valueTypes : indexTypes).push_back(std::move(arguments.additionalInputTypes[input]));
    }

    const auto& shape = logical->getShape();
    std::vector<uint64_t> indexDimensions(shape.begin(), shape.begin() + static_cast<std::ptrdiff_t>(numberOfIndices));
    return std::make_shared<TensorAggAggregationPhysicalFunction>(
        std::move(arguments.resultType),
        std::move(arguments.resultFieldIdentifier),
        std::move(valueFunctions),
        std::move(valueTypes),
        std::move(indexFunctions),
        std::move(indexTypes),
        std::move(indexDimensions),
        logical->getDefaultValue());
}
}
