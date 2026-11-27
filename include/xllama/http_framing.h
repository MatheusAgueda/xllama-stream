// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// HTTP/1.1 request framing for the LAN API (ADR 0006, audit F02). WinRT-free so
// the host suite can test it; the UWP transport only moves bytes and enforces
// the deadlines below.
#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>

namespace xllama {

inline constexpr size_t kHttpMaxHead = 64 * 1024;       // request line + headers
inline constexpr size_t kHttpMaxBody = 8 * 1024 * 1024; // larger is 413, not clamped
// Measured on Series S (2026-10-14): an 8 MiB body arrives in 1.3–1.4 s over
// the LAN. The body deadline is about 10× that, enough for a ~5 Mbit/s client;
// the head is a few hundred bytes. Past either, the request is 408.
inline constexpr std::chrono::seconds kHttpHeadDeadline{10};
inline constexpr std::chrono::seconds kHttpBodyDeadline{15};

// Replies have a deadline too. A client that stops reading fills the TCP window,
// and an unbounded store never returns: a streamed chat then holds the compute
// permit and its connection slot for as long as the peer stays connected
// (audit 2026-11-25, risk 3). Each write gets a 30 s floor plus 256 KiB/s for
// its bytes, so a 40 MB upscale reply to a slow client still fits (~190 s).
inline constexpr std::chrono::seconds kHttpWriteFloor{30};
inline constexpr size_t kHttpWriteMinRate = 256 * 1024; // bytes per second

inline std::chrono::milliseconds http_write_deadline(size_t bytes) {
    return kHttpWriteFloor + std::chrono::milliseconds(static_cast<long long>(
                                 bytes / kHttpWriteMinRate * 1000 +
                                 bytes % kHttpWriteMinRate * 1000 / kHttpWriteMinRate));
}

// What the connection's work item does with the socket once it has served the request
// (uwp/api-server.cpp, dispatch_connection). Every reply write, streamed tokens included, runs on
// that one thread-pool thread, so the thread keeps the mark.
//   begin()        a new connection starts on this thread (pool threads are reused: a mark left by
//                  an earlier stalled client must not close the next, healthy one)
//   expired()      a reply write passed http_write_deadline and was cancelled
//   close_socket() whether to Close() explicitly. A cancelled store leaves the connection open
//                  with no FIN for as long as the stalled peer is watched (console 2026-11-26,
//                  commit 013d403), so after a deadline nothing else ends it. Only then: on every
//                  other reply the release closes gracefully, and an explicit Close with a read
//                  still pending turns a 408 into a reset (1.6.0.1234).
class ReplyDeadlineMark {
  public:
    void begin() noexcept {
        hit_ = false;
    }
    void expired() noexcept {
        hit_ = true;
    }
    bool close_socket() const noexcept {
        return hit_;
    }

  private:
    bool hit_ = false;
};

struct HttpHead {
    // 0 when the head is valid; otherwise the HTTP status to answer with
    // (400, 411, 413 or 431) and `error` says why.
    int status = 0;
    std::string error;
    std::string method;
    std::string path; // query string removed
    std::string authorization;
    size_t content_length = 0;
    size_t body_start = 0; // offset of the body in the bytes parsed
};

// Offset just past "\r\n\r\n", or npos while the head is incomplete.
size_t http_head_end(std::string_view data);

// Parses the request line and headers of `data`, which must contain the whole
// head. Exact framing: complete header names (case-insensitive), one decimal
// Content-Length (repeats must agree), no Transfer-Encoding, and a body no
// larger than `max_body`.
HttpHead parse_http_head(std::string_view data, size_t max_body = kHttpMaxBody);

} // namespace xllama
