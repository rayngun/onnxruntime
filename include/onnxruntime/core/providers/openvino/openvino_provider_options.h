// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/// <summary>
/// Options for the OpenVINO provider that are passed to SessionOptionsAppendExecutionProvider_OpenVINO_V2.
/// Please note that this struct is *similar* to OrtOpenVINOProviderOptions but only to be used internally.
/// Going forward, new trt provider options are to be supported via this struct and usage of the publicly defined
/// OrtOpenVINOProviderOptions will be deprecated over time.
/// User can only get the instance of OrtOpenVINOProviderOptionsV2 via CreateOpenVINOProviderOptions.
/// </summary>
struct OrtOpenVINOProviderOptionsV2 {
  const char* device_type;
  unsigned char enable_vpu_fast_compile;  ///< 0 = disabled, nonzero = enabled
  const char* device_id;
  size_t num_of_threads;               ///< 0 = Use default number of threads
  unsigned char use_compiled_network;  ///< 0 = disabled, nonzero = enabled
  const char* blob_dump_path;          // path is set to empty by default
  void* context;
  unsigned char enable_opencl_throttling;  ///< 0 = disabled, nonzero = enabled
  unsigned char enable_dynamic_shapes;     ///< 0 = disabled, nonzero = enabled
};
