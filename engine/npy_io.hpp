#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

inline void parse_shape(const std::string& header, std::vector<size_t>& shape) {
    const auto pos = header.find("'shape'");
    if (pos == std::string::npos) throw std::runtime_error("npy: missing shape");
    auto lb = header.find('(', pos);
    auto rb = header.find(')', lb);
    if (lb == std::string::npos || rb == std::string::npos) throw std::runtime_error("npy: bad shape");
    std::string inner = header.substr(lb + 1, rb - lb - 1);
    shape.clear();
    std::stringstream ss(inner);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        shape.push_back(static_cast<size_t>(std::stoul(tok)));
    }
}

template <typename T>
std::vector<T> load(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("npy: bad magic");

    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl = 0;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else if (major == 2) {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("npy: unsupported version");
    }

    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    parse_shape(header, shape);

    std::vector<T> data;
    size_t count = 1;
    for (size_t d : shape) count *= d;
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!f) throw std::runtime_error("npy: read truncated");
    return data;
}

template <typename T>
void save(const std::string& path, const T* data, const std::vector<size_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot write " + path);
    f.write("\x93NUMPY", 6);
    uint8_t v1 = 1, v0 = 0;
    f.write(reinterpret_cast<const char*>(&v1), 1);
    f.write(reinterpret_cast<const char*>(&v0), 1);

    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) hdr << ", ";
        hdr << shape[i];
    }
    if (shape.size() == 1) hdr << ",";
    hdr << "), }";
    std::string header = hdr.str();
    // pad to 16-byte alignment with space + newline before closing paren area
    while ((10 + header.size()) % 16 != 0) header.push_back(' ');
    header.push_back('\n');
    uint16_t hl = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));

    size_t count = 1;
    for (size_t d : shape) count *= d;
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}

}  // namespace npy