#ifndef SSNSDP_READ_MODEL
#define SSNSDP_READ_MODEL

#include "core.h"

#include <filesystem>

#include "read_sdpa.h"
#include "read_ssnsdp.h"

namespace ssnsdp {
    SSNSDP_WEAK_INLINE
    Model read_model(const std::filesystem::path& path) {
        const auto& ext = path.extension();

        if (ext == ".dat-s") {
            return read_sdpa_from_file(path);
        }

        if (ext == ".ssnsdp") {
            return read_ssnsdp(path);
        }

        SSNSDP_ASSERT_MSG(false, "Only .dat-s and .ssnsdp files are supported");
        return Model();
    }
}
#endif
