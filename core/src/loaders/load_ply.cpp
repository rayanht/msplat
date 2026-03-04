#include "loaders.hpp"
#include <fstream>
#include <sstream>
#include <cstring>

// Property types in PLY files
enum class PlyType { Float32, Float64, UInt8, UInt16, Int32, Unknown };

static PlyType parsePlyType(const std::string &t) {
    if (t == "float" || t == "float32") return PlyType::Float32;
    if (t == "double" || t == "float64") return PlyType::Float64;
    if (t == "uchar" || t == "uint8") return PlyType::UInt8;
    if (t == "ushort" || t == "uint16") return PlyType::UInt16;
    if (t == "int" || t == "int32") return PlyType::Int32;
    return PlyType::Unknown;
}

static size_t plyTypeSize(PlyType t) {
    switch (t) {
        case PlyType::Float32: return 4;
        case PlyType::Float64: return 8;
        case PlyType::UInt8:   return 1;
        case PlyType::UInt16:  return 2;
        case PlyType::Int32:   return 4;
        default: return 0;
    }
}

struct PlyProp {
    std::string name;
    PlyType type;
    int offset;     // byte offset within a vertex record
};

Points readPly(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open PLY file: " + path);

    std::string line;
    std::getline(f, line);
    if (line.find("ply") == std::string::npos)
        throw std::runtime_error("Not a PLY file: " + path);

    bool binary = false;
    bool ascii = false;
    int numVertices = 0;
    std::vector<PlyProp> props;
    int vertexBytes = 0;

    while (std::getline(f, line)) {
        if (line == "end_header") break;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string fmt;
            iss >> fmt;
            binary = (fmt == "binary_little_endian");
            ascii = (fmt == "ascii");
        } else if (token == "element") {
            std::string name;
            iss >> name;
            if (name == "vertex") iss >> numVertices;
        } else if (token == "property") {
            // Only collect vertex properties (before any other element)
            if (numVertices > 0) {
                std::string typeStr, name;
                iss >> typeStr >> name;
                PlyType t = parsePlyType(typeStr);
                props.push_back({name, t, vertexBytes});
                vertexBytes += (int)plyTypeSize(t);
            }
        }
    }

    if (numVertices == 0) throw std::runtime_error("PLY has no vertices");
    if (!binary && !ascii) throw std::runtime_error("Unsupported PLY format (need ascii or binary_little_endian)");

    // Find property indices
    auto findProp = [&](const std::string &name) -> int {
        for (int i = 0; i < (int)props.size(); i++)
            if (props[i].name == name) return i;
        return -1;
    };

    int ix = findProp("x"), iy = findProp("y"), iz = findProp("z");
    int ir = findProp("red"), ig = findProp("green"), ib = findProp("blue");
    if (ix < 0 || iy < 0 || iz < 0)
        throw std::runtime_error("PLY missing x/y/z properties");

    Points pts;
    pts.count = numVertices;
    pts.xyz.resize(numVertices * 3);
    pts.rgb.resize(numVertices * 3, 128); // default gray if no color

    if (binary) {
        std::vector<char> buf(vertexBytes);
        for (int i = 0; i < numVertices; i++) {
            f.read(buf.data(), vertexBytes);

            auto readFloat = [&](int pi) -> float {
                const auto &p = props[pi];
                if (p.type == PlyType::Float32) { float v; memcpy(&v, &buf[p.offset], 4); return v; }
                if (p.type == PlyType::Float64) { double v; memcpy(&v, &buf[p.offset], 8); return (float)v; }
                return 0.0f;
            };
            auto readUint8 = [&](int pi) -> uint8_t {
                const auto &p = props[pi];
                if (p.type == PlyType::UInt8) return (uint8_t)buf[p.offset];
                if (p.type == PlyType::UInt16) { uint16_t v; memcpy(&v, &buf[p.offset], 2); return (uint8_t)(v >> 8); }
                if (p.type == PlyType::Float32) { float v; memcpy(&v, &buf[p.offset], 4); return (uint8_t)(v * 255.0f); }
                return 128;
            };

            pts.xyz[i*3+0] = readFloat(ix);
            pts.xyz[i*3+1] = readFloat(iy);
            pts.xyz[i*3+2] = readFloat(iz);

            if (ir >= 0 && ig >= 0 && ib >= 0) {
                pts.rgb[i*3+0] = readUint8(ir);
                pts.rgb[i*3+1] = readUint8(ig);
                pts.rgb[i*3+2] = readUint8(ib);
            }
        }
    } else {
        // ASCII
        for (int i = 0; i < numVertices; i++) {
            std::getline(f, line);
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string tok;
            while (iss >> tok) tokens.push_back(tok);

            pts.xyz[i*3+0] = std::stof(tokens[ix]);
            pts.xyz[i*3+1] = std::stof(tokens[iy]);
            pts.xyz[i*3+2] = std::stof(tokens[iz]);

            if (ir >= 0 && ig >= 0 && ib >= 0) {
                float rv = std::stof(tokens[ir]);
                float gv = std::stof(tokens[ig]);
                float bv = std::stof(tokens[ib]);
                // Detect if values are 0-1 float or 0-255 int
                if (props[ir].type == PlyType::Float32 || props[ir].type == PlyType::Float64) {
                    pts.rgb[i*3+0] = (uint8_t)(rv * 255.0f);
                    pts.rgb[i*3+1] = (uint8_t)(gv * 255.0f);
                    pts.rgb[i*3+2] = (uint8_t)(bv * 255.0f);
                } else {
                    pts.rgb[i*3+0] = (uint8_t)rv;
                    pts.rgb[i*3+1] = (uint8_t)gv;
                    pts.rgb[i*3+2] = (uint8_t)bv;
                }
            }
        }
    }

    // point count available via pts.count
    return pts;
}

// COLMAP points3D.bin reader
Points readColmapPoints(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open points3D.bin: " + path);

    uint64_t numPoints;
    f.read(reinterpret_cast<char*>(&numPoints), 8);

    Points pts;
    pts.count = (int64_t)numPoints;
    pts.xyz.resize(numPoints * 3);
    pts.rgb.resize(numPoints * 3);

    for (uint64_t i = 0; i < numPoints; i++) {
        uint64_t pointId;
        f.read(reinterpret_cast<char*>(&pointId), 8);

        double x, y, z;
        f.read(reinterpret_cast<char*>(&x), 8);
        f.read(reinterpret_cast<char*>(&y), 8);
        f.read(reinterpret_cast<char*>(&z), 8);
        pts.xyz[i*3+0] = (float)x;
        pts.xyz[i*3+1] = (float)y;
        pts.xyz[i*3+2] = (float)z;

        uint8_t r, g, b;
        f.read(reinterpret_cast<char*>(&r), 1);
        f.read(reinterpret_cast<char*>(&g), 1);
        f.read(reinterpret_cast<char*>(&b), 1);
        pts.rgb[i*3+0] = r;
        pts.rgb[i*3+1] = g;
        pts.rgb[i*3+2] = b;

        double error;
        f.read(reinterpret_cast<char*>(&error), 8);

        uint64_t trackLen;
        f.read(reinterpret_cast<char*>(&trackLen), 8);
        f.seekg(trackLen * 8, std::ios::cur); // skip track entries (imageId u32 + point2dIdx u32 each)
    }

    // point count available via pts.count
    return pts;
}
