#ifndef SSNSDP_READ_SDPA_H
#define SSNSDP_READ_SDPA_H

#include "core.h"

#include <filesystem>
#include <algorithm>
#include <string>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "model.h"
#include "cell_array.h"

namespace ssnsdp {

    // read data from sdpa formatted string in memory
    // the contents of the string may be changed
    // Use C-style IO for better performance
    SSNSDP_WEAK_INLINE
    Model read_sdpa_sparse(std::string& input) {

        using std::strchr;
        using std::atoi;
        using std::strtol;
        using std::strtod;

        Model model;

        // preprocess: remove special chars
        const Size str_new_size = std::remove_if(input.begin(), input.end(),
            [](const unsigned char x) {
                return x == '(' || x == ')' || x == ',' || x == '{' || x == '}';
            }) - input.begin();
        input.resize(str_new_size);

        const char* cur_cstr = input.data();
        const char* endl_ptr;
        // string_view input_view(input.c_str(), str_new_size);
        // size_type next_line_pos;
        // istringstream s_in(input);

        // first line
        // string line;
        // getline(s_in, line);

        // is title and comments ?
        if (*cur_cstr == '*' || *cur_cstr == '"') {
            // ignore it, get the next line
            endl_ptr = strchr(cur_cstr, '\n');
            SSNSDP_ASSERT(endl_ptr);
            cur_cstr = endl_ptr + 1;
        }

        // mDIM, nBLOCK and bLOCKsTRUCT

        // line example:    3  =  mDIM
        endl_ptr = strchr(cur_cstr, '\n');
        SSNSDP_ASSERT(endl_ptr);
        const auto m = atoi(cur_cstr);
        cur_cstr = endl_ptr + 1;

        // line example:    1  =  nBLOCK
        endl_ptr = strchr(cur_cstr, '\n');
        SSNSDP_ASSERT(endl_ptr);
        const auto n_block = atoi(cur_cstr);
        cur_cstr = endl_ptr + 1;

        // line example:    2  = bLOCKsTRUCT
        endl_ptr = strchr(cur_cstr, '\n');
        SSNSDP_ASSERT(endl_ptr);
        CellArray<BlockSpec>& blk = model.blk();
        blk.resize(n_block);

        // cast away const for strto* function
        char* endptr = const_cast<char*>(cur_cstr);
        for (Index p = 0; p < n_block; ++p) {
            BlockSpec& pblk = blk[p];
            pblk.n = strtol(endptr, &endptr, 10);
            pblk.type = pblk.n > 1 ? BlockType::SDP : BlockType::LINEAR;
            pblk.n = abs(pblk.n);
        }
        cur_cstr = endl_ptr + 1;

        // No comment in the input from now on.
        // Therefore we do not have to deal with line end.

        // read constant vector b
        // line example: 48, -8, 20
        VectorX& b = model.b();
        b.resize(m);
        endptr = const_cast<char*>(cur_cstr);
        for (Index i = 0; i < m; ++i) {
            const auto elem = strtod(endptr, &endptr);
            const Scalar real_elem = as_scalar(-elem);
            b(i) = real_elem;
        }

        // read constant matrixes
        CellArray<std::vector<TripletList>> Fs_elems(n_block);

        for (Size p = 0; p < n_block; ++p) {
            if (blk[p].is_linear()) {
                Fs_elems[p] = std::vector<TripletList>(2);
            } else {
                Fs_elems[p] = std::vector<TripletList>(m + 1);
            }
        }

        // line example: 0 1 1 1 -11
        while (true) {
            // sscanf is slow! Read each number individually.
            const auto mat_index = strtol(endptr, &endptr, 10);
            const auto blk_index = strtol(endptr, &endptr, 10) - 1;
            const auto ii = strtol(endptr, &endptr, 10);
            const auto jj = strtol(endptr, &endptr, 10);
            cur_cstr = endptr;
            const auto elem = strtod(endptr, &endptr);

            if (endptr == cur_cstr) {
                break;
            }
            cur_cstr = endptr;
            // In release mode where assertion is disabled,
            //   behaviour is undefined and may lead to security problem.
            // Only use trusted input!
            SSNSDP_ASSERT_MSG(mat_index <= m, "Index out of range");

            const Scalar real_elem = as_scalar(-elem);

            // mat_index == 0: C
            // mat_index > 0: At[mat_index]
            if (blk[blk_index].is_linear()) {
                SSNSDP_ASSERT_MSG(ii == jj, "At[k] for linear block should be diagonal matrix.");
                if (mat_index == 0) {
                    Fs_elems[blk_index][0].emplace_back(ii - 1, 0, real_elem);
                } else {
                    Fs_elems[blk_index][1].emplace_back(ii - 1, mat_index - 1, real_elem);
                }
            } else {
                Fs_elems[blk_index][mat_index].emplace_back(ii - 1, jj - 1, real_elem);
                // `At' is processed with svec that only requires Upper part,
                // while C should be stored entirely.
                if (mat_index == 0 && ii != jj) {
                    Fs_elems[blk_index][0].emplace_back(jj - 1, ii - 1, real_elem);
                }
            }
        }

        CellArray<SparseMatrix>& C = model.C();
        C.resize(n_block);
        CellArray<SparseMatrix>& At = model.At();
        At.resize(n_block);

        for (Index p = 0; p < n_block; ++p) {
            const auto& Fs_elems_p = Fs_elems[p];
            const Size np = blk[p].n;

            if (blk[p].is_linear()) {
                C[p] = SparseMatrix(np, 1);
                At[p] = SparseMatrix(np, m);
                At[p].setFromTriplets(Fs_elems_p[1].begin(), Fs_elems_p[1].end());
            } else {
                C[p] = SparseMatrix(np, np);
                At[p] = SparseMatrix(np * (np + 1) / 2, m);
                for (Index i = 1; i <= m; ++i) {
                    SparseMatrix F(np, np);
                    F.setFromTriplets(Fs_elems_p[i].begin(), Fs_elems_p[i].end());
                    At[p].col(i - 1) = svec_sparse(F, F.nonZeros());
                }
                // At[p] is already compressed.
            }

            C[p].setFromTriplets(Fs_elems_p[0].begin(), Fs_elems_p[0].end());
        }

        return model;
    }

    // read data from a file
    SSNSDP_WEAK_INLINE
    Model read_sdpa_from_file(const std::filesystem::path& path) {
        namespace fs = std::filesystem;
        // path ends with .dat-s
        SSNSDP_ASSERT_MSG(path.extension() == ".dat-s",
            "Other file extensions than `.dat-s' are not supported");
        SSNSDP_ASSERT_MSG(fs::exists(path), "File not exist");
        const Size file_size = fs::file_size(path);
        // read whole input into a string
        std::string buffer(file_size, '\0');
        std::ifstream fin(path, std::ifstream::binary);
        fin.read(buffer.data(), file_size);
        fin.close();

        return read_sdpa_sparse(buffer);
    }
}

#endif
