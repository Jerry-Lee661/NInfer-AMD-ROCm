#include "hip_bf16_rename.h"
#pragma once
// HIP backend shim: route CUB block primitives to hipCUB (rocPRIM backend).
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
