// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/compute_admission.h"
#include "xllama/withdrawn_models.h"

#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace xllama {

struct PullModelDescriptor {
    std::string name;
    std::string kind;
    std::string role;
    std::string source_url;
    bool trusted_catalogue = false;
    std::vector<std::string> sha256_pins;
};

inline std::string resolve_pull_model_name(const std::string& name) {
    if (name == "bge-m3")
        return "embed-bge-m3";
    if (name == "nomic-embed-text-v2-moe")
        return "embed-nomic-v2-moe";
    if (name == "qwen3-embedding:4b")
        return "embed-qwen3-4b";
    return name;
}

inline bool valid_pull_sha256(const std::string& pin) {
    if (pin.size() != 64)
        return false;
    for (const char c : pin) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    }
    return true;
}

inline bool api_pull_model_allowed(const PullModelDescriptor& model) {
    if (!model.trusted_catalogue || model.name.empty() ||
        model.source_url.rfind("https://", 0) != 0 || model.sha256_pins.empty())
        return false;
    if (model.kind != "gguf" && model.kind != "ort-genai")
        return false;
    if (model.role == "diffusion" || model.role == "image")
        return false;
    if (model.role == "embedding" && model.kind != "gguf")
        return false;
    // Defence in depth: /api/pull reads the bundled list today; the same
    // withdrawn list is dropped from LocalState overrides in LoadModelManifest.
    if (model_id_is_withdrawn(model.name))
        return false;
    if (!model.role.empty() && model.role != "coding" && model.role != "embedding")
        return false;
    for (const auto& pin : model.sha256_pins) {
        if (!valid_pull_sha256(pin))
            return false;
    }
    return true;
}

class ApiPullGate {
  public:
    std::unique_lock<std::mutex> try_acquire() {
        return std::unique_lock<std::mutex>(m_mutex, std::try_to_lock);
    }

  private:
    std::mutex m_mutex;
};

// POST /api/pull admission: the pull gate, then the compute permit, BEFORE the reply stream opens
// or a byte is downloaded. Busy is 503 with nothing written; a second pull is 409. The permit is
// held through the download and the model load (the pull needs the slot at the end anyway and
// used to find it taken after the whole download), and the caller releases it before its final
// reply write.
enum class PullAdmitStatus { Admitted, PullInProgress, Busy };

struct PullAdmission {
    PullAdmitStatus status = PullAdmitStatus::PullInProgress;
    std::unique_lock<std::mutex> gate;
    ComputeJobAdmission::Handle permit;

    void release() {
        permit.reset();
        if (gate.owns_lock())
            gate.unlock();
    }
};

template <typename AcquirePermit>
PullAdmission admit_pull(ApiPullGate& gate, AcquirePermit&& acquire_permit) {
    PullAdmission a;
    a.gate = gate.try_acquire();
    if (!a.gate.owns_lock())
        return a;
    a.permit = acquire_permit();
    if (!a.permit) {
        a.gate.unlock();
        a.status = PullAdmitStatus::Busy;
        return a;
    }
    a.status = PullAdmitStatus::Admitted;
    return a;
}

} // namespace xllama
