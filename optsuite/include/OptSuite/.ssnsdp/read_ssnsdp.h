#ifndef SSNSDP_READ_SSNSDP
#define SSNSDP_READ_SSNSDP

#include "core.h"

#include <fstream>
#include <filesystem>
#include <string_view>
#include <cstdint>

#include "model.h"

namespace ssnsdp {

    SSNSDP_WEAK_INLINE
    Model read_ssnsdp(std::istream& input) {
        using namespace std::string_view_literals;
        Model model;

        // use std::endian when C++20 is supported
        constexpr std::uint32_t one_for_endianness_test = 1;
        SSNSDP_ASSERT_MSG(*reinterpret_cast<const char*>(&one_for_endianness_test) == 1,
            "Little endianness is assumed to use .ssnsdp file");

        constexpr auto int_size = sizeof(uint32_t);
        union {
            std::uint32_t i;
            char s[8];
        } tmp;

        input.read(tmp.s, 8);
        [[maybe_unused]] const std::string_view file_header(tmp.s, 8);
        SSNSDP_ASSERT(file_header == "SSNSDP01"sv);

        input.read(reinterpret_cast<char*>(&tmp.i), int_size);
        const Size m = as_int(tmp.i);
        input.read(reinterpret_cast<char*>(&tmp.i), int_size);
        const Size nblock = as_int(tmp.i);

        auto& blk = model.blk();
        blk.resize(nblock);
        for (Index k = 0; k < nblock; ++k) {
            input.read(reinterpret_cast<char*>(&tmp.i), int_size);
            blk[k].type = static_cast<BlockType>(tmp.i);
            input.read(reinterpret_cast<char*>(&tmp.i), int_size);
            blk[k].n = as_int(tmp.i);
        }

        auto& b = model.b();
        b.resize(m);
        input.read(reinterpret_cast<char*>(b.data()), sizeof(double) * m);

        std::vector<uint32_t> row_indices;
        std::vector<uint32_t> col_indices;
        std::vector<double> values;
        TripletList triplet_list;
        auto& nnz = tmp.i;

        auto& C = model.C();
        C.resize(nblock);
        for (Index k = 0; k < nblock; ++k) {
            const auto& kblk = blk[k];

            input.read(reinterpret_cast<char*>(&nnz), int_size);
            row_indices.resize(nnz);
            values.resize(nnz);
            input.read(reinterpret_cast<char*>(row_indices.data()), int_size * nnz);
            if (kblk.is_sdp()) {
                col_indices.resize(nnz);
                triplet_list.clear();
                triplet_list.reserve(nnz * 2);
                input.read(reinterpret_cast<char*>(col_indices.data()), int_size * nnz);
                input.read(reinterpret_cast<char*>(values.data()), sizeof(double) * nnz);

                for (Index kk = 0; kk < nnz; ++kk) {
                    const auto i = static_cast<SparseIndex>(row_indices[kk]);
                    const auto j = static_cast<SparseIndex>(col_indices[kk]);
                    const auto v = as_scalar(values[kk]);
                    triplet_list.emplace_back(i, j, v);
                    if (i != j) {
                        triplet_list.emplace_back(j, i, v);
                    }
                }
                C[k] = SparseMatrix(kblk.n, kblk.n);
                C[k].setFromTriplets(triplet_list.begin(), triplet_list.end());
            } else {
                SSNSDP_ASSUME(kblk.is_linear());
                input.read(reinterpret_cast<char*>(values.data()), sizeof(double) * nnz);

                SparseVector Ck(kblk.n);
                Ck.reserve(nnz);

                for (Index kk = 0; kk < nnz; ++kk) {
                    const auto i = static_cast<SparseIndex>(row_indices[kk]);
                    const auto v = as_scalar(values[kk]);
                    Ck.insert(i) = v;
                }
                C[k] = Ck;
            }
        }

        auto& At = model.At();
        At.resize(nblock);
        for (Index k = 0; k < nblock; ++k) {
            const auto& kblk = blk[k];

            input.read(reinterpret_cast<char*>(&nnz), int_size);
            row_indices.resize(nnz);
            col_indices.resize(nnz);
            values.resize(nnz);
            triplet_list.resize(nnz);

            input.read(reinterpret_cast<char*>(row_indices.data()), int_size * nnz);
            input.read(reinterpret_cast<char*>(col_indices.data()), int_size * nnz);
            input.read(reinterpret_cast<char*>(values.data()), sizeof(double) * nnz);

            const auto n = kblk.n;
            if (kblk.is_sdp()) {
                At[k] = SparseMatrix(n * (n + 1) / 2, m);
            } else {
                SSNSDP_ASSUME(kblk.is_linear());
                At[k] = SparseMatrix(n, m);
            }

            for (Index kk = 0; kk < nnz; ++kk) {
                const auto i = static_cast<SparseIndex>(row_indices[kk]);
                const auto j = static_cast<SparseIndex>(col_indices[kk]);
                const auto v = as_scalar(values[kk]);
                triplet_list[kk] = Triplet(i, j, v);
            }

            At[k].setFromTriplets(triplet_list.begin(), triplet_list.end());
        }

        return model;
    }

    SSNSDP_WEAK_INLINE
    Model read_ssnsdp(const std::filesystem::path& path) {
        namespace fs = std::filesystem;
        SSNSDP_ASSERT_MSG(path.extension() == ".ssnsdp",
            "Other file extensions than `.ssnsdp' are not supported");
        SSNSDP_ASSERT_MSG(fs::exists(path), "File not exist");
        std::ifstream fin(path, std::ifstream::binary);
        Model model = read_ssnsdp(fin);
        fin.close();
        return model;
    }
}

#endif
