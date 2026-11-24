#include "aconvnet_engine.hpp"

#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <unordered_map>

#include "npy_io.hpp"
#include "round.hpp"

namespace {

std::unordered_map<std::string, float> parse_act_scales(const std::string& text) {
    std::unordered_map<std::string, float> out;
    std::regex re(R"re("(\w+)"\s*:\s*([0-9.eE+-]+))re");
    const auto sub = text.substr(text.find("\"act_scales\""));
    auto begin = std::sregex_iterator(sub.begin(), sub.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const std::string key = (*it)[1].str();
        if (key == "act_scales") continue;
        out[key] = std::stof((*it)[2].str());
    }
    return out;
}

int parse_int_field(const std::string& text, const std::string& key) {
    std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) throw std::runtime_error("manifest missing " + key);
    return std::stoi(m[1].str());
}

LayerParams load_layer(const std::string& dir, const std::string& name) {
    LayerParams p;
    std::vector<size_t> shape_w, shape_sw, shape_b;
    p.weight_q = npy::load<int8_t>(dir + "/" + name + "/weight_q.npy", shape_w);
    p.scale_w = npy::load<float>(dir + "/" + name + "/scale_w.npy", shape_sw);
    p.bias_fp = npy::load<float>(dir + "/" + name + "/bias_fp.npy", shape_b);
    if (shape_w.size() != 4) throw std::runtime_error("bad weight shape");
    p.out_c = static_cast<int>(shape_w[0]);
    p.in_c = static_cast<int>(shape_w[1]);
    p.kh = static_cast<int>(shape_w[2]);
    p.kw = static_cast<int>(shape_w[3]);
    return p;
}

void max_pool2x2(const float* in, int c, int h, int w, float* out) {
    const int ho = h / 2;
    const int wo = w / 2;
    for (int ic = 0; ic < c; ++ic) {
        for (int i = 0; i < ho; ++i) {
            for (int j = 0; j < wo; ++j) {
                float m = -1e30f;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const float v = in[ic * h * w + (i * 2 + dy) * w + (j * 2 + dx)];
                        m = std::max(m, v);
                    }
                }
                out[ic * ho * wo + i * wo + j] = m;
            }
        }
    }
}

void conv2d_fp_valid(
    const float* x, int xc, int xh, int xw,
    const LayerParams& lp, float* y) {
    const int out_c = lp.out_c;
    const int kh = lp.kh;
    const int kw = lp.kw;
    const int yh = xh - kh + 1;
    const int yw = xw - kw + 1;

    std::vector<float> w_deq(lp.weight_q.size());
    for (size_t i = 0; i < lp.weight_q.size(); ++i) {
        const int oc = static_cast<int>(i / (lp.in_c * kh * kw));
        const float sw = lp.scale_w.size() > 1 ? lp.scale_w[oc] : lp.scale_w[0];
        w_deq[i] = quant::dequantize(lp.weight_q[i], sw);
    }

    for (int oc = 0; oc < out_c; ++oc) {
        const float sw = lp.scale_w.size() > 1 ? lp.scale_w[oc] : lp.scale_w[0];
        (void)sw;
        for (int i = 0; i < yh; ++i) {
            for (int j = 0; j < yw; ++j) {
                float acc = lp.bias_fp[oc];
                for (int ic = 0; ic < xc; ++ic) {
                    for (int dy = 0; dy < kh; ++dy) {
                        for (int dx = 0; dx < kw; ++dx) {
                            const size_t widx =
                                (static_cast<size_t>(oc) * lp.in_c + ic) * kh * kw + dy * kw + dx;
                            const float xv = x[ic * xh * xw + (i + dy) * xw + (j + dx)];
                            acc += xv * w_deq[widx];
                        }
                    }
                }
                y[oc * yh * yw + i * yw + j] = acc;
            }
        }
    }
}

void conv2d_int32_valid(
    const int8_t* x, int xc, int xh, int xw, float scale_in,
    const LayerParams& lp, float* y_fp) {
    const int out_c = lp.out_c;
    const int kh = lp.kh;
    const int kw = lp.kw;
    const int yh = xh - kh + 1;
    const int yw = xw - kw + 1;

    for (int oc = 0; oc < out_c; ++oc) {
        const float sw = lp.scale_w.size() > 1 ? lp.scale_w[oc] : lp.scale_w[0];
        const float mult = scale_in * sw;
        for (int i = 0; i < yh; ++i) {
            for (int j = 0; j < yw; ++j) {
                int32_t acc = 0;
                for (int ic = 0; ic < xc; ++ic) {
                    for (int dy = 0; dy < kh; ++dy) {
                        for (int dx = 0; dx < kw; ++dx) {
                            const size_t widx =
                                (static_cast<size_t>(oc) * lp.in_c + ic) * kh * kw + dy * kw + dx;
                            const int8_t xv = x[ic * xh * xw + (i + dy) * xw + (j + dx)];
                            acc += static_cast<int32_t>(xv) * static_cast<int32_t>(lp.weight_q[widx]);
                        }
                    }
                }
                y_fp[oc * yh * yw + i * yw + j] = static_cast<float>(acc) * mult + lp.bias_fp[oc];
            }
        }
    }
}

void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = std::max(0.f, x[i]);
}

void quantize_map(const float* x, int n, float scale, int8_t* q) {
    for (int i = 0; i < n; ++i) q[i] = quant::quantize_symmetric(x[i], scale);
}

}  // namespace

EnginePackage load_package(const std::string& root) {
    std::ifstream mf(root + "/manifest.json");
    if (!mf) throw std::runtime_error("missing manifest.json");
    std::string text((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    EnginePackage pkg;
    pkg.in_channels = parse_int_field(text, "in_channels");
    pkg.num_classes = parse_int_field(text, "num_classes");
    pkg.act_scales = parse_act_scales(text);
    for (const char* name : {"conv1", "conv2", "conv3", "conv4", "conv5"}) {
        pkg.layers[name] = load_layer(root, name);
    }
    return pkg;
}

std::vector<float> forward_weight_only(const EnginePackage& pkg, const float* input_nchw, int n) {
    const int ic0 = pkg.in_channels;
    const int ih0 = 88;
    const int iw0 = 88;

    std::vector<float> logits(static_cast<size_t>(n) * pkg.num_classes);
    std::vector<float> a, b, c, d, e;

    for (int batch = 0; batch < n; ++batch) {
        const float* in = input_nchw + static_cast<size_t>(batch) * ic0 * ih0 * iw0;
        a.resize(static_cast<size_t>(16) * 84 * 84);
        conv2d_fp_valid(in, ic0, ih0, iw0, pkg.layers.at("conv1"), a.data());
        relu_inplace(a.data(), static_cast<int>(a.size()));
        b.resize(static_cast<size_t>(16) * 42 * 42);
        max_pool2x2(a.data(), 16, 84, 84, b.data());

        c.resize(static_cast<size_t>(32) * 38 * 38);
        conv2d_fp_valid(b.data(), 16, 42, 42, pkg.layers.at("conv2"), c.data());
        relu_inplace(c.data(), static_cast<int>(c.size()));
        d.resize(static_cast<size_t>(32) * 19 * 19);
        max_pool2x2(c.data(), 32, 38, 38, d.data());

        e.resize(static_cast<size_t>(64) * 14 * 14);
        conv2d_fp_valid(d.data(), 32, 19, 19, pkg.layers.at("conv3"), e.data());
        relu_inplace(e.data(), static_cast<int>(e.size()));
        std::vector<float> f(static_cast<size_t>(64) * 7 * 7);
        max_pool2x2(e.data(), 64, 14, 14, f.data());

        std::vector<float> g(static_cast<size_t>(128) * 3 * 3);
        conv2d_fp_valid(f.data(), 64, 7, 7, pkg.layers.at("conv4"), g.data());
        relu_inplace(g.data(), static_cast<int>(g.size()));

        std::vector<float> h(static_cast<size_t>(pkg.num_classes) * 1 * 1);
        conv2d_fp_valid(g.data(), 128, 3, 3, pkg.layers.at("conv5"), h.data());

        for (int k = 0; k < pkg.num_classes; ++k) {
            logits[static_cast<size_t>(batch) * pkg.num_classes + k] = h[k];
        }
    }
    return logits;
}

std::vector<float> forward_full_int8(const EnginePackage& pkg, const float* input_nchw, int n) {
    std::vector<float> logits(static_cast<size_t>(n) * pkg.num_classes);

    for (int batch = 0; batch < n; ++batch) {
        const float* in = input_nchw + static_cast<size_t>(batch) * pkg.in_channels * 88 * 88;
        const float s_in = pkg.act_scales.at("in");
        std::vector<int8_t> xq(static_cast<size_t>(pkg.in_channels) * 88 * 88);
        quantize_map(in, static_cast<int>(xq.size()), s_in, xq.data());
        float s_act = s_in;
        int xc = pkg.in_channels, xh = 88, xw = 88;

        auto run_block = [&](const std::string& lname, bool pool) {
            std::vector<float> y_fp(static_cast<size_t>(pkg.layers.at(lname).out_c) * (xh - pkg.layers.at(lname).kh + 1) *
                                    (xw - pkg.layers.at(lname).kw + 1));
            const int yh = xh - pkg.layers.at(lname).kh + 1;
            const int yw = xw - pkg.layers.at(lname).kw + 1;
            conv2d_int32_valid(xq.data(), xc, xh, xw, s_act, pkg.layers.at(lname), y_fp.data());
            relu_inplace(y_fp.data(), static_cast<int>(y_fp.size()));
            if (pool) {
                xc = pkg.layers.at(lname).out_c;
                xh = yh / 2;
                xw = yw / 2;
                std::vector<float> pooled(static_cast<size_t>(xc) * xh * xw);
                max_pool2x2(y_fp.data(), pkg.layers.at(lname).out_c, yh, yw, pooled.data());
                y_fp = std::move(pooled);
            } else {
                xc = pkg.layers.at(lname).out_c;
                xh = yh;
                xw = yw;
            }
            s_act = pkg.act_scales.at(lname);
            xq.resize(static_cast<size_t>(xc) * xh * xw);
            quantize_map(y_fp.data(), static_cast<int>(xq.size()), s_act, xq.data());
        };

        run_block("conv1", true);
        run_block("conv2", true);
        run_block("conv3", true);
        run_block("conv4", false);

        std::vector<float> logit(static_cast<size_t>(pkg.num_classes));
        conv2d_int32_valid(xq.data(), xc, xh, xw, s_act, pkg.layers.at("conv5"), logit.data());
        for (int k = 0; k < pkg.num_classes; ++k) {
            logits[static_cast<size_t>(batch) * pkg.num_classes + k] = logit[k];
        }
    }
    return logits;
}