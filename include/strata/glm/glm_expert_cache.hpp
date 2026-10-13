// include/strata/glm/glm_expert_cache.hpp - GLM-5.3-Flash's routed experts streamed from the GGUF through a cache of
// blob slots, for a model whose experts do not all fit in memory (the UD-IQ4_XS file is 157 GB; this machine has
// 128 GB).
//
// The slots hold the same [gate rows | up rows | down rows] blobs the resident path builds (native_expert_layout), in
// device memory or in mapped host memory, and a per-block device table of n_expert addresses points the grouped
// expert kernel at them.  After a MoE layer's router (GlmExpertProvider::experts) the chosen ids come to the host -
// one stream synchronization a MoE layer, which also means no kernel still reads a slot about to be refilled - and:
//   - an expert in a slot is a hit (the slot moves to the back of the LRU list)
//   - a miss takes the least recently used slot that this call's experts do not hold, and is copied there from the
//     mapped file (its three slabs, from the stacked gate / up / down tensors), the misses of a call in parallel
//     threads; then the block's table is uploaded
// Before the first token the cache can be filled (fill_all): as many experts of every block as fit, block by block.
//
// Measured on the Radeon 8060S (Strix Halo, Windows; glm_loader_parity --bench), at GLM-5.3-Flash's expert size
// (IQ4_XS, 13.4 MB a blob), from a file whose pages are in the OS cache:
//   a call's 8 misses into mapped host slots: 3.6 ms (30 GB/s, the same for 1, 4 or 8 threads); into device slots
//   8.1 ms with 4 threads (13 GB/s; 3.3 GB/s with one: cudaMemcpy from pageable memory)
//   the grouped expert kernel on 8 experts: 0.48 ms from device slots (220 GB/s), 0.56 ms from mapped host slots
//   (190 GB/s) - on an APU both are the same DRAM, and mapped host memory is not limited to the device carve-out
//   a call whose 8 experts all hit: 29-33 us (the ids' trip and the synchronization), 1.3-1.4 ms over 43 MoE layers
// A miss from a cold file waits for the drive (13.4 MB at a few GB/s: several ms).
//
// What it is not yet: no prefetch (a miss waits for the file), no CPU compute of misses, LRU only (no frequency
// profile for the fill), and the copy goes through the OS's mapping of the file (the page cache), not unbuffered reads.
#pragma once

#include "strata/glm/glm_layer.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::glm {

/// One MoE block's experts in the mapped file: the stacked tensors' data and the blob layout they repack into.
struct GlmExpertSource {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    kernels::NativeExpertLayout layout;
};

class GlmExpertCache : public GlmExpertProvider {
public:
    struct Stats {
        uint64_t calls = 0, hits = 0, misses = 0, bytes = 0;
        double sync_ms = 0;    ///< waiting for the router's ids
        double fetch_ms = 0;   ///< copying misses into slots, the table upload included
    };

    GlmExpertCache() = default;
    ~GlmExpertCache() override;
    GlmExpertCache(const GlmExpertCache&) = delete;
    GlmExpertCache& operator=(const GlmExpertCache&) = delete;

    /// `sources` by block (a dense block's empty); `slots` blobs of the largest layout, in device memory or mapped
    /// host memory (`mapped_host`); `threads` copies at once.  The mapped file must outlive the cache.
    bool init(std::vector<GlmExpertSource> sources, int64_t n_expert, int64_t slots, bool mapped_host, int threads,
              std::string& err);
    /// Fills the free slots with experts of every MoE block, an equal share each, in expert order.
    bool fill_all(std::string& err);

    const unsigned long long* experts(int64_t il, const int32_t* ids, int k, void* stream, std::string& err) override;

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }
    int64_t slots() const { return (int64_t) owner_.size(); }
    uint64_t slot_bytes() const { return slot_bytes_; }
    int64_t resident() const;   ///< slots holding an expert

private:
    std::vector<GlmExpertSource> src_;
    int64_t n_expert_ = 0;
    uint64_t slot_bytes_ = 0;
    bool mapped_host_ = false;
    int threads_ = 1;
    uint8_t* host_ = nullptr;     ///< mapped host slots (the CPU's address), or null
    uint8_t* blobs_ = nullptr;    ///< the slots' base address as the GPU reads it
    unsigned long long* tables_ = nullptr;   ///< device [blocks][n_expert]
    std::vector<unsigned long long> host_tables_;
    int32_t* ids_host_ = nullptr;            ///< pinned, for the router's ids
    std::vector<int32_t> slot_of_;           ///< [blocks][n_expert]: the slot, or -1
    std::vector<int64_t> owner_;             ///< slot -> block * n_expert + expert, or -1
    std::vector<int32_t> prev_, next_;       ///< the LRU list over slots: head_ least recent
    int32_t head_ = -1, tail_ = -1;
    std::vector<uint64_t> used_;             ///< slot -> the call that last used it
    uint64_t call_ = 0;
    Stats stats_;

    void unlink(int32_t s);
    void push_back(int32_t s);
    bool fetch(const std::vector<std::pair<int64_t, int32_t>>& jobs, std::string& err);   ///< (owner, slot)
    void release();
};

}  // namespace strata::glm
