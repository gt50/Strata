// src/glm/glm_expert_cache.cpp - see include/strata/glm/glm_expert_cache.hpp.
#include "strata/glm/glm_expert_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

namespace strata::glm {
namespace {

uint64_t align256(uint64_t n) { return (n + 255) & ~(uint64_t) 255; }

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    cudaGetLastError();
    err = std::string("GlmExpertCache: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

GlmExpertCache::~GlmExpertCache() { release(); }

void GlmExpertCache::release() {
    if (host_) cudaFreeHost(host_);
    else if (blobs_) cudaFree(blobs_);
    if (tables_) cudaFree(tables_);
    if (ids_host_) cudaFreeHost(ids_host_);
    host_ = blobs_ = nullptr;
    tables_ = nullptr;
    ids_host_ = nullptr;
}

bool GlmExpertCache::init(std::vector<GlmExpertSource> sources, int64_t n_expert, int64_t slots, bool mapped_host,
                          int threads, std::string& err) {
    release();
    src_ = std::move(sources);
    n_expert_ = n_expert;
    mapped_host_ = mapped_host;
    threads_ = std::max(1, threads);
    slot_bytes_ = 0;
    for (const GlmExpertSource& s : src_)
        if (s.gate) slot_bytes_ = std::max<uint64_t>(slot_bytes_, align256(s.layout.bytes));
    if (slot_bytes_ == 0 || n_expert <= 0 || slots <= 0) {
        err = "GlmExpertCache: no MoE block, no experts, or no slots";
        return false;
    }
    const uint64_t total = slot_bytes_ * (uint64_t) slots;
    if (mapped_host) {
        if (!cuda_ok(cudaHostAlloc((void**) &host_, total, cudaHostAllocMapped), "the slots (mapped host)", err) ||
            !cuda_ok(cudaHostGetDevicePointer((void**) &blobs_, host_, 0), "the slots' device address", err)) {
            release();
            return false;
        }
    } else if (!cuda_ok(cudaMalloc((void**) &blobs_, total), "the slots (device)", err)) {
        release();
        return false;
    }
    const size_t nb = src_.size();
    if (!cuda_ok(cudaMalloc((void**) &tables_, nb * (size_t) n_expert * 8), "the tables", err) ||
        !cuda_ok(cudaHostAlloc((void**) &ids_host_, 1024 * sizeof(int32_t), cudaHostAllocDefault), "the ids", err)) {
        release();
        return false;
    }
    // absent experts point at slot 0: never read (only the chosen ids are, and those are resident), but valid
    host_tables_.assign(nb * (size_t) n_expert, (unsigned long long) (uintptr_t) blobs_);
    if (!cuda_ok(cudaMemcpy(tables_, host_tables_.data(), host_tables_.size() * 8, cudaMemcpyHostToDevice), "tables",
                 err)) {
        release();
        return false;
    }
    slot_of_.assign(nb * (size_t) n_expert, -1);
    owner_.assign((size_t) slots, -1);
    used_.assign((size_t) slots, 0);
    prev_.assign((size_t) slots, -1);
    next_.assign((size_t) slots, -1);
    head_ = tail_ = -1;
    for (int32_t s = 0; s < (int32_t) slots; ++s) push_back(s);
    call_ = 0;
    stats_ = Stats{};
    return true;
}

int64_t GlmExpertCache::resident() const {
    return (int64_t) std::count_if(owner_.begin(), owner_.end(), [](int64_t o) { return o >= 0; });
}

void GlmExpertCache::unlink(int32_t s) {
    if (prev_[(size_t) s] >= 0) next_[(size_t) prev_[(size_t) s]] = next_[(size_t) s];
    else head_ = next_[(size_t) s];
    if (next_[(size_t) s] >= 0) prev_[(size_t) next_[(size_t) s]] = prev_[(size_t) s];
    else tail_ = prev_[(size_t) s];
    prev_[(size_t) s] = next_[(size_t) s] = -1;
}

void GlmExpertCache::push_back(int32_t s) {
    prev_[(size_t) s] = tail_;
    next_[(size_t) s] = -1;
    if (tail_ >= 0) next_[(size_t) tail_] = s;
    else head_ = s;
    tail_ = s;
}

/// Copies each job's expert (owner = block * n_expert + expert) from the mapped file into its slot: the three
/// slabs of the stacked tensors, the jobs spread over threads_ threads.
bool GlmExpertCache::fetch(const std::vector<std::pair<int64_t, int32_t>>& jobs, std::string& err) {
    if (jobs.empty()) return true;
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::string first_err;
    auto work = [&]() {
        for (size_t j; (j = next.fetch_add(1)) < jobs.size();) {
            const int64_t block = jobs[j].first / n_expert_, e = jobs[j].first % n_expert_;
            const GlmExpertSource& s = src_[(size_t) block];
            const kernels::NativeExpertLayout& el = s.layout;
            const uint64_t gu = el.up_off, dn = el.bytes - el.down_off, off = slot_bytes_ * (uint64_t) jobs[j].second;
            const uint8_t* g = s.gate + gu * (uint64_t) e;
            const uint8_t* u = s.up + gu * (uint64_t) e;
            const uint8_t* d = s.down + dn * (uint64_t) e;
            if (host_) {
                std::memcpy(host_ + off, g, gu);
                std::memcpy(host_ + off + el.up_off, u, gu);
                std::memcpy(host_ + off + el.down_off, d, dn);
            } else {
                uint8_t* b = blobs_ + off;
                if (cudaMemcpy(b, g, gu, cudaMemcpyHostToDevice) != cudaSuccess ||
                    cudaMemcpy(b + el.up_off, u, gu, cudaMemcpyHostToDevice) != cudaSuccess ||
                    cudaMemcpy(b + el.down_off, d, dn, cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (!failed.exchange(true)) first_err = "GlmExpertCache: copying an expert into its slot failed";
                    return;
                }
            }
        }
    };
    const int n = (int) std::min<size_t>((size_t) threads_, jobs.size());
    if (n <= 1) work();
    else {
        std::vector<std::thread> ts;
        for (int i = 0; i < n; ++i) ts.emplace_back(work);
        for (auto& t : ts) t.join();
    }
    if (failed) {
        err = first_err;
        return false;
    }
    for (const auto& [owner, slot] : jobs) stats_.bytes += src_[(size_t) (owner / n_expert_)].layout.bytes;
    return true;
}

bool GlmExpertCache::fill_all(std::string& err) {
    std::vector<int64_t> moe;
    for (size_t b = 0; b < src_.size(); ++b)
        if (src_[b].gate) moe.push_back((int64_t) b);
    std::vector<int32_t> free_slots;
    for (int32_t s = 0; s < (int32_t) owner_.size(); ++s)
        if (owner_[(size_t) s] < 0) free_slots.push_back(s);
    const int64_t per = std::min<int64_t>(n_expert_, (int64_t) free_slots.size() / (int64_t) moe.size());
    std::vector<std::pair<int64_t, int32_t>> jobs;
    size_t f = 0;
    for (int64_t b : moe)
        for (int64_t e = 0; e < per; ++e) {
            const int64_t o = b * n_expert_ + e;
            if (slot_of_[(size_t) o] < 0) jobs.emplace_back(o, free_slots[f++]);
        }
    if (!fetch(jobs, err)) return false;
    for (const auto& [o, s] : jobs) {
        owner_[(size_t) s] = o;
        slot_of_[(size_t) o] = s;
        host_tables_[(size_t) o] = (unsigned long long) (uintptr_t) (blobs_ + slot_bytes_ * (uint64_t) s);
    }
    stats_.bytes = 0;   // the fill is not a token's cost
    return cuda_ok(cudaMemcpy(tables_, host_tables_.data(), host_tables_.size() * 8, cudaMemcpyHostToDevice), "tables",
                   err);
}

const unsigned long long* GlmExpertCache::experts(int64_t il, const int32_t* ids, int k, void* stream,
                                                   std::string& err) {
    if (!tables_ || il < 0 || il >= (int64_t) src_.size() || !src_[(size_t) il].gate || k <= 0 || k > 1024) {
        err = "GlmExpertCache: block " + std::to_string(il) + " has no streamed experts, or k is out of range";
        return nullptr;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const auto t0 = std::chrono::steady_clock::now();
    // the router's choice; the synchronization also means no kernel before it still reads a slot
    if (!cuda_ok(cudaMemcpyAsync(ids_host_, ids, (size_t) k * sizeof(int32_t), cudaMemcpyDeviceToHost, st), "ids",
                 err) ||
        !cuda_ok(cudaStreamSynchronize(st), "the router", err))
        return nullptr;
    stats_.sync_ms += ms_since(t0);
    return admit(il, ids_host_, k, stream, err);
}

const unsigned long long* GlmExpertCache::resident(int64_t il, const int32_t* experts, int n, void* stream,
                                                   std::string& err) {
    if (!tables_ || il < 0 || il >= (int64_t) src_.size() || !src_[(size_t) il].gate || n <= 0 ||
        n > (int) owner_.size()) {
        err = "GlmExpertCache: block " + std::to_string(il) + " has no streamed experts, or " + std::to_string(n) +
              " experts are more than the " + std::to_string(owner_.size()) + " slots";
        return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!cuda_ok(cudaStreamSynchronize((cudaStream_t) stream), "the round before", err)) return nullptr;
    stats_.sync_ms += ms_since(t0);
    return admit(il, experts, n, stream, err);
}

const unsigned long long* GlmExpertCache::admit(int64_t il, const int32_t* ids_host, int k, void* stream,
                                                std::string& err) {
    const cudaStream_t st = (cudaStream_t) stream;
    const auto t0 = std::chrono::steady_clock::now();
    ++stats_.calls;
    const uint64_t call = ++call_;
    std::vector<int64_t> missing;
    for (int i = 0; i < k; ++i) {
        const int32_t e = ids_host[i];
        if (e < 0 || e >= n_expert_) {
            err = "GlmExpertCache: the router chose expert " + std::to_string(e);
            return nullptr;
        }
        const int64_t o = il * n_expert_ + e;
        const int32_t s = slot_of_[(size_t) o];
        if (s >= 0) {
            ++stats_.hits;
            used_[(size_t) s] = call;
            unlink(s);
            push_back(s);
        } else if (std::find(missing.begin(), missing.end(), o) == missing.end()) {
            missing.push_back(o);
        }
    }
    if (missing.empty()) return tables_ + (size_t) il * (size_t) n_expert_;
    std::vector<std::pair<int64_t, int32_t>> jobs;
    for (int64_t o : missing) {
        int32_t s = head_;   // the least recent slot this call does not hold
        while (s >= 0 && used_[(size_t) s] == call) s = next_[(size_t) s];
        if (s < 0) {
            err = "GlmExpertCache: " + std::to_string(owner_.size()) + " slots cannot hold one layer's " +
                  std::to_string(k) + " experts";
            return nullptr;
        }
        if (owner_[(size_t) s] >= 0) slot_of_[(size_t) owner_[(size_t) s]] = -1;
        owner_[(size_t) s] = o;
        slot_of_[(size_t) o] = s;
        used_[(size_t) s] = call;
        unlink(s);
        push_back(s);
        host_tables_[(size_t) o] = (unsigned long long) (uintptr_t) (blobs_ + slot_bytes_ * (uint64_t) s);
        jobs.emplace_back(o, s);
    }
    stats_.misses += jobs.size();
    if (!fetch(jobs, err)) return nullptr;
    // the block's table, from a pageable vector: staged before the call returns, so the next change cannot race it
    unsigned long long* t = tables_ + (size_t) il * (size_t) n_expert_;
    if (!cuda_ok(cudaMemcpyAsync(t, host_tables_.data() + (size_t) il * (size_t) n_expert_, (size_t) n_expert_ * 8,
                                 cudaMemcpyHostToDevice, st),
                 "table", err))
        return nullptr;
    stats_.fetch_ms += ms_since(t0);
    return t;
}

}  // namespace strata::glm
