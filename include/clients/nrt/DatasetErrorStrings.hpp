#pragma once

namespace fluid
{
namespace client
{
  constexpr auto  NoBufferError = "No buffer passed";
  constexpr auto  NotFoundError = "Point not found";
  constexpr auto  WrongSizeError = "Wrong Point Size";
  constexpr auto  WrongInitError = "Wrong number of initial points";
  constexpr auto  DuplicateError = "Label already in dataset";
  constexpr auto  SmallDatasetError = "Dataset is smaller than k";
}
}
