// Presence of this file makes Next.js fall back to Babel instead of its
// native SWC compiler (`@next/swc-linux-x64-gnu`), whose prebuilt binary
// assumes AVX2 CPU support and crashes with an illegal-instruction fault
// on older hardware. See frontend/README-avx2-fallback.md.
//
// Trade-off: Babel is pure JS (works on any CPU) but noticeably slower to
// compile than SWC, and a few SWC-only optimizations are unavailable.
module.exports = {
  presets: ["next/babel"],
};
