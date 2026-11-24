#include <cstring>
#include <iostream>
#include <string>

#include "aconvnet_engine.hpp"
#include "npy_io.hpp"

int main(int argc, char** argv) {
    std::string package = "runs/quant_chen/int8_package";
    std::string input = "runs/quant_chen/int_ref_verify/layer_dump_sample0/input.npy";
    std::string output;
    std::string mode = "weight_only";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--package") == 0 && i + 1 < argc) package = argv[++i];
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i];
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
    }
    if (output.empty()) {
        std::cerr << "usage: aconvnet_int8 --package DIR --input IN.npy --output OUT.npy "
                     "[--mode weight_only|full_int8]\n";
        return 1;
    }

    std::vector<size_t> shape;
    auto data = npy::load<float>(input, shape);
    if (shape.size() != 4) {
        std::cerr << "expected input shape (N,C,H,W)\n";
        return 1;
    }
    const int n = static_cast<int>(shape[0]);

    try {
        const EnginePackage pkg = load_package(package);
        std::vector<float> logits;
        if (mode == "full_int8") logits = forward_full_int8(pkg, data.data(), n);
        else logits = forward_weight_only(pkg, data.data(), n);

        const std::vector<size_t> out_shape{shape[0], static_cast<size_t>(pkg.num_classes)};
        npy::save(output, logits.data(), out_shape);
        std::cout << "wrote " << output << " mode=" << mode << " N=" << n << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}