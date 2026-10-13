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
// From a cold file, through the mapping (each 4 KB page faulted in), the real UD-IQ4_XS model's misses came at 0.45-0.9
// GB/s on a Samsung 990 EVO Plus: nearly all of a token's time.  So when the cache knows the files (init's `files`),
// a miss is read unbuffered instead (platform::DirectFile: FILE_FLAG_NO_BUFFERING / O_DIRECT, many reads in flight):
// its three slabs, each widened to 4 KB boundaries, into one of a ring of pinned staging buffers, then copied to its
// slot on a copy stream while the next reads run.  The mapped path stays for a cache without files (and for A/B:
// STRATA_GLM_MAPPED_READS=1).  The real file then: 2.3-2.6 GB/s of misses (the drive: 3.6-4.0 GB/s unbuffered at 3.5 MB
// reads, 8-32 in flight), a 24-token prompt 61.9 -> 17.2 s, decode 0.76 -> 1.92 tokens/s (docs/GLM.md).  From a
// warm file the mapping is faster (13 GB/s into device slots against 5.5 for unbuffered reads, --bench), but the
// model's experts are three times this PC's RAM.
//
// What it is not yet: no prefetch (a miss waits for the file), no CPU compute of misses, LRU only (no frequency
// profile for the fill).
#pragma once

#include "strata/glm/glm_layer.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/platform/direct_file.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

/// One MoE block's experts in the mapped file: the stacked tensors' data and the blob layout they repack into; and
/// for unbuffered reads, each tensor's file (an index into init's `files`, -1: none) and offset in it.
struct GlmExpertSource {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    kernels::NativeExpertLayout layout;
    int gate_file = -1, up_file = -1, down_file = -1;
    uint64_t gate_off = 0, up_off = 0, down_off = 0;
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
    /// host memory (`mapped_host`); `threads` copies at once.  The mapped file must outlive the cache.  `files`: the
    /// paths the sources' file indices name; when given (and every source has its files), misses are read unbuffered.
    bool init(std::vector<GlmExpertSource> sources, int64_t n_expert, int64_t slots, bool mapped_host, int threads,
              std::string& err, const std::vector<std::string>& files = {});
    /// Fills the free slots with experts of every MoE block, an equal share each, in expert order.
    bool fill_all(std::string& err);

    const unsigned long long* experts(int64_t il, const int32_t* ids, int k, void* stream, std::string& err) override;
    const unsigned long long* resident(int64_t il, const int32_t* experts, int n, void* stream,
                                       std::string& err) override;
    int64_t capacity() const override { return (int64_t) owner_.size(); }
    const uint8_t* slot_base() const override { return blobs_; }
    uint64_t slot_stride() const override { return slot_bytes_; }
    int32_t slot_of(int64_t il, int32_t e) const override { return slot_of_[(size_t) (il * n_expert_ + e)]; }

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }
    int64_t slots() const { return (int64_t) owner_.size(); }
    uint64_t slot_bytes() const { return slot_bytes_; }
    int64_t resident() const;   ///< slots holding an expert
    bool direct() const { return !files_.empty(); }   ///< misses read unbuffered (not through the mapping)

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
    // unbuffered reads: the files, a ring of pinned staging buffers (stage_bytes_ each), an event each, a copy stream
    std::vector<std::unique_ptr<platform::DirectFile>> files_;
    uint8_t* staging_ = nullptr;
    uint64_t stage_bytes_ = 0;
    std::vector<void*> stage_events_;
    void* copy_stream_ = nullptr;

    void unlink(int32_t s);
    void push_back(int32_t s);
    bool fetch(const std::vector<std::pair<int64_t, int32_t>>& jobs, std::string& err);   ///< (owner, slot)
    bool fetch_direct(const std::vector<std::pair<int64_t, int32_t>>& jobs, std::string& err);
    /// The hits and misses of host ids[0, k) in block il, the stream synchronized before
    const unsigned long long* admit(int64_t il, const int32_t* ids, int k, void* stream, std::string& err);
    void release();
};

}  // namespace strata::glm
