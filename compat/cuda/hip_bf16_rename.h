#pragma once
// Renames the handful of amd_hip_bf16.h functions whose parameters are plain scalars
// (float/double/int/ushort/short/float2) - those are the only bf16 exports that cannot
// coexist with compat's own overloads as legal C++ overloads. Everything taking a bf16
// type (__hip_bfloat16/__hip_bfloat162 vs our __nv_bfloat16/__nv_bfloat162) is already
// a distinct parameter list and overloads cleanly, so those names are left untouched.
#define __float2bfloat16      ninfer_hipref_float2bfloat16
#define __double2bfloat16     ninfer_hipref_double2bfloat16
#define __int2bfloat16_rn     ninfer_hipref_int2bfloat16_rn
#define __ushort_as_bfloat16  ninfer_hipref_ushort_as_bfloat16
#define __short_as_bfloat16   ninfer_hipref_short_as_bfloat16
#define __float22bfloat162_rn ninfer_hipref_float22bfloat162_rn
#define __h2div               ninfer_hipref_h2div

// ---- Self-contained activation -------------------------------------------------
// Include the HIP headers under the renames, then immediately drop the FUNCTION
// name mappings so that any call site anywhere later in the include chain resolves
// to compat's own CUDA-compatible definitions in cuda_bf16.h. The type mappings
// stay active for the rest of the translation unit on purpose: the HIP headers
// never define the original type names (they were rewritten), so undefining them
// would break later references.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#undef __float2bfloat16
#undef __double2bfloat16
#undef __int2bfloat16_rn
#undef __ushort_as_bfloat16
#undef __short_as_bfloat16
#undef __float22bfloat162_rn
#undef __h2div
#undef __bfloat162float
#undef __bfloat1622float2
#undef __bfloat16_as_ushort
#undef __low2bfloat16
#undef __high2bfloat16
#undef __low2float
#undef __high2float
#undef __halves2bfloat162
#undef __bfloat162bfloat162
#undef __floats2bfloat162_rn
#undef __hadd2
#undef __hsub2
#undef __hmul2
#undef __hsub2_rn
#undef __habs2
