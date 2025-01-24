// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <map>
#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace openvino_ep {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(ProviderInfo provider_info, SharedContext& shared_context)
      : provider_info_(provider_info), shared_context_(shared_context) {}

  ~OpenVINOProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<OpenVINOExecutionProvider>(provider_info_, shared_context_);
  }

 private:
  ProviderInfo provider_info_;
  SharedContext& shared_context_;
};

struct ProviderInfo_OpenVINO_Impl : ProviderInfo_OpenVINO {
  std::vector<std::string> GetAvailableDevices() const override {
    return OVCore::GetAvailableDevices();
  }
};

struct OpenVINO_Provider : Provider {
  void* GetInfo() override { return &info_; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    // Extract the void_params into ProviderOptions and ConfigOptions
    using ConfigBuffer = std::pair<const ProviderOptions*, const ConfigOptions&>;
    const ConfigBuffer* buffer = reinterpret_cast<const ConfigBuffer*>(void_params);
    const auto& provider_options_map = *buffer->first;
    const auto& config_options = buffer->second;

    ProviderInfo pi;

    std::string bool_flag = "";
    if (provider_options_map.find("device_type") != provider_options_map.end()) {
      pi.device_type = provider_options_map.at("device_type").c_str();

      std::set<std::string> ov_supported_device_types = {"CPU", "GPU",
                                                         "GPU.0", "GPU.1", "NPU"};
      std::set<std::string> deprecated_device_types = {"CPU_FP32", "GPU_FP32",
                                                       "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                       "GPU.0_FP16", "GPU.1_FP16"};
      std::vector<std::string> available_devices = OVCore::GetAvailableDevices();

      for (auto& device : available_devices) {
        if (ov_supported_device_types.find(device) == ov_supported_device_types.end()) {
          ov_supported_device_types.emplace(device);
        }
      }
      if (deprecated_device_types.find(pi.device_type) != deprecated_device_types.end()) {
        std::string deprecated_device = pi.device_type;
        int delimit = pi.device_type.find("_");
        pi.device_type = deprecated_device.substr(0, delimit);
        pi.precision = deprecated_device.substr(delimit + 1);
        LOGS_DEFAULT(WARNING) << "[OpenVINO] Selected 'device_type' " + deprecated_device + " is deprecated. \n"
                              << "Update the 'device_type' to specified types 'CPU', 'GPU', 'GPU.0', "
                              << "'GPU.1', 'NPU' or from"
                              << " HETERO/MULTI/AUTO options and set 'precision' separately. \n";
      }
      if (!((ov_supported_device_types.find(pi.device_type) != ov_supported_device_types.end()) ||
            (pi.device_type.find("HETERO:") == 0) ||
            (pi.device_type.find("MULTI:") == 0) ||
            (pi.device_type.find("AUTO:") == 0))) {
        ORT_THROW(
            "[ERROR] [OpenVINO] You have selected wrong configuration value for the key 'device_type'. "
            "Select from 'CPU', 'GPU', 'NPU', 'GPU.x' where x = 0,1,2 and so on or from"
            " HETERO/MULTI/AUTO options available. \n");
      }
    }
    if (provider_options_map.find("device_id") != provider_options_map.end()) {
      std::string dev_id = provider_options_map.at("device_id").c_str();
      LOGS_DEFAULT(WARNING) << "[OpenVINO] The options 'device_id' is deprecated. "
                            << "Upgrade to set deice_type and precision session options.\n";
      if (dev_id == "CPU" || dev_id == "GPU" || dev_id == "NPU") {
        pi.device_type = std::move(dev_id);
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported device_id is selected. Select from available options.");
      }
    }
    if (provider_options_map.find("precision") != provider_options_map.end()) {
      pi.precision = provider_options_map.at("precision").c_str();
    }
    if (pi.device_type.find("GPU") != std::string::npos) {
      if (pi.precision == "") {
        pi.precision = "FP16";
      } else if (pi.precision != "ACCURACY" && pi.precision != "FP16" && pi.precision != "FP32") {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. GPU only supports FP32 / FP16. \n");
      }
    } else if (pi.device_type.find("NPU") != std::string::npos) {
      if (pi.precision == "" || pi.precision == "ACCURACY" || pi.precision == "FP16") {
        pi.precision = "FP16";
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. NPU only supported FP16. \n");
      }
    } else if (pi.device_type.find("CPU") != std::string::npos) {
      if (pi.precision == "" || pi.precision == "ACCURACY" || pi.precision == "FP32") {
        pi.precision = "FP32";
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. CPU only supports FP32 . \n");
      }
    }

    if (provider_options_map.find("cache_dir") != provider_options_map.end()) {
      pi.cache_dir = provider_options_map.at("cache_dir");
    }

    if (provider_options_map.find("load_config") != provider_options_map.end()) {
      auto parse_config = [&](const std::string& config_str) -> std::map<std::string, ov::AnyMap> {
        // If the config string is empty, return an empty map and skip processing
        if (config_str.empty()) {
          LOGS_DEFAULT(WARNING) << "Empty OV Config Map passed. Skipping load_config option parsing.\n";
          return {};
        }

        std::stringstream input_str_stream(config_str);
        std::map<std::string, ov::AnyMap> target_map;

        try {
          nlohmann::json json_config = nlohmann::json::parse(input_str_stream);

          if (!json_config.is_object()) {
            ORT_THROW("Invalid JSON structure: Expected an object at the root.");
          }

          for (auto& [key, value] : json_config.items()) {
            ov::AnyMap inner_map;

            // Ensure the key is one of "CPU", "GPU", or "NPU"
            if (key != "CPU" && key != "GPU" && key != "NPU") {
              LOGS_DEFAULT(WARNING) << "Unsupported device key: " << key << ". Skipping entry.\n";
              continue;
            }

            // Ensure that the value for each device is an object (PROPERTY -> VALUE)
            if (!value.is_object()) {
              ORT_THROW("Invalid JSON structure: Expected an object for device properties.");
            }

            for (auto& [inner_key, inner_value] : value.items()) {
              if (inner_value.is_string()) {
                inner_map[inner_key] = inner_value.get<std::string>();
              } else if (inner_value.is_number_integer()) {
                inner_map[inner_key] = inner_value.get<int64_t>();
              } else if (inner_value.is_number_float()) {
                inner_map[inner_key] = inner_value.get<double>();
              } else if (inner_value.is_boolean()) {
                inner_map[inner_key] = inner_value.get<bool>();
              } else {
                LOGS_DEFAULT(WARNING) << "Unsupported JSON value type for key: " << inner_key << ". Skipping key.";
              }
            }
            target_map[key] = std::move(inner_map);
          }
        } catch (const nlohmann::json::parse_error& e) {
          // Handle syntax errors in JSON
          ORT_THROW("JSON parsing error: " + std::string(e.what()));
        } catch (const nlohmann::json::type_error& e) {
          // Handle invalid type accesses
          ORT_THROW("JSON type error: " + std::string(e.what()));
        } catch (const std::exception& e) {
          ORT_THROW("Error parsing load_config Map: " + std::string(e.what()));
        }
        return target_map;
      };

      pi.load_config = parse_config(provider_options_map.at("load_config"));
    }

    if (provider_options_map.find("context") != provider_options_map.end()) {
      std::string str = provider_options_map.at("context");
      uint64_t number = std::strtoull(str.c_str(), nullptr, 16);
      pi.context = reinterpret_cast<void*>(number);
    }
#if defined(IO_BUFFER_ENABLED)
    // a valid context must be provided to enable IO Buffer optimizations
    if (context == nullptr) {
#undef IO_BUFFER_ENABLED
#define IO_BUFFER_ENABLED = 0
      LOGS_DEFAULT(WARNING) << "Context is not set. Disabling IO Buffer optimization";
    }
#endif

    if (provider_options_map.find("num_of_threads") != provider_options_map.end()) {
      if (!std::all_of(provider_options_map.at("num_of_threads").begin(),
                       provider_options_map.at("num_of_threads").end(), ::isdigit)) {
        ORT_THROW("[ERROR] [OpenVINO-EP] Number of threads should be a number. \n");
      }
      pi.num_of_threads = std::stoi(provider_options_map.at("num_of_threads"));
      if (pi.num_of_threads <= 0) {
        pi.num_of_threads = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_threads' should be in the positive range.\n "
                              << "Executing with num_threads=1";
      }
    }

    if (provider_options_map.find("model_priority") != provider_options_map.end()) {
      pi.model_priority = provider_options_map.at("model_priority").c_str();
      std::vector<std::string> supported_priorities({"LOW", "MEDIUM", "HIGH", "DEFAULT"});
      if (std::find(supported_priorities.begin(), supported_priorities.end(),
                    pi.model_priority) == supported_priorities.end()) {
        pi.model_priority = "DEFAULT";
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'model_priority' "
                              << "is not one of LOW, MEDIUM, HIGH, DEFAULT. "
                              << "Executing with model_priorty=DEFAULT";
      }
    }

    if (provider_options_map.find("num_streams") != provider_options_map.end()) {
      pi.num_streams = std::stoi(provider_options_map.at("num_streams"));
      if (pi.num_streams <= 0) {
        pi.num_streams = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_streams' should be in the range of 1-8.\n "
                              << "Executing with num_streams=1";
      }
    }
    if (provider_options_map.find("enable_opencl_throttling") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_opencl_throttling");
      if (bool_flag == "true" || bool_flag == "True")
        pi.enable_opencl_throttling = true;
      else if (bool_flag == "false" || bool_flag == "False")
        pi.enable_opencl_throttling = false;
      bool_flag = "";
    }

    if (provider_options_map.find("enable_qdq_optimizer") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_qdq_optimizer");
      if (bool_flag == "true" || bool_flag == "True")
        pi.enable_qdq_optimizer = true;
      else if (bool_flag == "false" || bool_flag == "False")
        pi.enable_qdq_optimizer = false;
      else
        ORT_THROW("[ERROR] [OpenVINO-EP] enable_qdq_optimiser should be a boolean.\n");
      bool_flag = "";
    }

    // Always true for NPU plugin or when passed .
    if (pi.device_type.find("NPU") != std::string::npos) {
      pi.disable_dynamic_shapes = true;
    }
    if (provider_options_map.find("disable_dynamic_shapes") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("disable_dynamic_shapes");
      if (bool_flag == "true" || bool_flag == "True") {
        pi.disable_dynamic_shapes = true;
      } else if (bool_flag == "false" || bool_flag == "False") {
        if (pi.device_type.find("NPU") != std::string::npos) {
          pi.disable_dynamic_shapes = true;
          LOGS_DEFAULT(INFO) << "[OpenVINO-EP] The value for the key 'disable_dynamic_shapes' will be set to "
                             << "TRUE for NPU backend.\n ";
        } else {
          pi.disable_dynamic_shapes = false;
        }
      }
      bool_flag = "";
    }

    pi.so_disable_cpu_ep_fallback = config_options.GetConfigOrDefault(kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";
    pi.so_context_enable = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
    pi.so_context_embed_mode = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
    pi.so_share_ep_contexts = config_options.GetConfigOrDefault(kOrtSessionOptionShareEpContexts, "0") == "1";
    pi.so_context_file_path = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");

    // Append values to config to support weight-as-inputs conversion for shared contexts
    if (pi.so_share_ep_contexts) {
      ov::AnyMap map;
      map["NPU_COMPILATION_MODE_PARAMS"] = "enable-wd-blockarg-input=true compute-layers-with-higher-precision=Sqrt,Power,ReduceSum";
      pi.load_config["NPU"] = map;
    }

    return std::make_shared<OpenVINOProviderFactory>(pi, shared_context_);
  }

  void Initialize() override {
    OVCore::Initialize();
  }

  void Shutdown() override {
    backend_utils::DestroyOVTensors(shared_context_.shared_weights.metadata);
    OVCore::Teardown();
  }

 private:
  SharedContext shared_context_;
  ProviderInfo_OpenVINO_Impl info_;
};  // OpenVINO_Provider

}  // namespace openvino_ep
}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  static onnxruntime::openvino_ep::OpenVINO_Provider g_provider;
  return &g_provider;
}
}
