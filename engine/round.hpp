#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace quant {

constexpr int32_t QMIN = -128;
constexpr int32_t QMAX = 127;

inline float round_half_to_even(float x) {
    return std::nearbyintf(x);
}

inline int8_t quantize_symmetric(float x, float scale) {
    const float qf = round_half_to_even(x / scale);
    int32_t q = static_cast<int32_t>(qf);
    if (qf > static_cast<float>(QMAX)) q = QMAX;
    if (qf < static_cast<float>(QMIN)) q = QMIN;
    return static_cast<int8_t>(q);
}

inline float dequantize(int8_t q, float scale) {
    return static_cast<float>(q) * scale;
}

}  // namespace quant