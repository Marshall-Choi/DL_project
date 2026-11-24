#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct LayerParams {
    std::vector<int8_t> weight_q;
    std::vector<float> scale_w;
    std::vector<float> bias_fp;
    int out_c = 0;
    int in_c = 0;
    int kh = 0;
    int kw = 0;
};

struct EnginePackage {
    int in_channels = 2;
    int num_classes = 10;
    std::unordered_map<std::string, float> act_scales;
    std::unordered_map<std::string, LayerParams> layers;
};

EnginePackage load_package(const std::string& root);

std::vector<float> forward_weight_only(const EnginePackage& pkg, const float* input_nchw, int n = 1);

std::vector<float> forward_full_int8(const EnginePackage& pkg, const float* input_nchw, int n = 1);