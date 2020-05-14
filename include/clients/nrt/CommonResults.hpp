#pragma once

#include <clients/common/Result.hpp>

namespace fluid {
namespace client {
static const MessageResult<void> OKResult = {Result::Status::kOk};
static const MessageResult<void> NoBufferError = {Result::Status::kError,
                                                  "No buffer passed"};
static const MessageResult<void> PointNotFoundError = {Result::Status::kError,
                                                       "Point not found"};
static const MessageResult<void> WrongPointSizeError = {Result::Status::kError,
                                                        "Wrong Point Size"};
static const MessageResult<void> WrongPointNumError = {
    Result::Status::kError, "Wrong number of points"};

static const MessageResult<void> WrongInitError = {
    Result::Status::kError, "Wrong number of initial points"};
static const MessageResult<void> DuplicateError = {Result::Status::kError,
                                                   "Label already in dataset"};
static const MessageResult<void> SmallDataSetError = {
    Result::Status::kError, "DataSet is smaller than k"};
static const MessageResult<void> SmallKError = {Result::Status::kError,
                                                "k is too small"};

static const MessageResult<void> EmptyDataSetError = {Result::Status::kError,
                                                      "DataSet is empty"};

static const MessageResult<void> EmptyLabelSetError = {Result::Status::kError,
                                                       "LabelSet is empty"};

static const MessageResult<void> NoDataSetError = {Result::Status::kError,
                                                   "DataSet does not exist"};
static const MessageResult<void> NoLabelSetError = {Result::Status::kError,
                                                    "LabelSet does not exist"};

static const MessageResult<void> NoDataSetOrLabelSetError = {
    Result::Status::kError, "Missing DataSet or LabelSet"};

static const MessageResult<void> NoDataFittedError = {Result::Status::kError,
                                                      "No data fitted"};

static const MessageResult<void> NotEnoughDataError = {Result::Status::kError,
                                                       "Not enough data"};
static const MessageResult<void> EmptyLabelError = {Result::Status::kError,
                                                    "Empty label"};
static const MessageResult<void> EmptyIdError = {Result::Status::kError,
                                                 "Empty id"};

static const MessageResult<void> BufferAllocError = {Result::Status::kError,
                                                     "Can't allocate buffer"};

static const MessageResult<void> ReadError = {Result::Status::kError,
                                              "Couldn't read file"};
static const MessageResult<void> WriteError = {Result::Status::kError,
                                               "Couldn't write file"};
static const MessageResult<void> NotImplementedError = {Result::Status::kError,
                                                        "Not implemented"};

} // namespace client
} // namespace fluid
