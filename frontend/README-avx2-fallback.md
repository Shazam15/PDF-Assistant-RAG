# AVX2-compatible frontend toolchain (branch: `frontend/tailwind-v3-no-avx2-fallback`)

## Why this branch exists

On CPUs without AVX2 (e.g. pre-Haswell Intel — Sandy Bridge, Ivy Bridge —
which includes machines like a Dell OptiPlex 790), `next dev` / `next build`
would crash-loop and eventually exhaust system memory. Root cause: two
pieces of the default frontend toolchain ship prebuilt **native Rust
binaries** that assume AVX2 on Linux x64:

- Next.js's compiler, `@next/swc-linux-x64-gnu`
- Tailwind CSS v4's engine, `@tailwindcss/oxide`

Without AVX2, invoking either binary faults with an illegal-instruction
crash. Something in the pipeline retried without backoff, so instead of a
single clean error, the machine saw hundreds of processes spawn per second
until the OOM killer started taking down unrelated processes (VS Code, in
the case that prompted this branch).

The **real target deployment** (Xeon Gold + Tesla T4, per the main
`README.md`) has a modern CPU with AVX2 and never hits this. This branch
exists purely so the frontend can also be smoke-tested on older/spare
hardware. It is not needed, and not recommended as the default, for the
actual T4 server.

## What changed

1. **Tailwind CSS v4 → v3.4** (`frontend/package.json`, `frontend/postcss.config.mjs`)
   v3 is pure JS/PostCSS — no native binary, no AVX2 dependency.

2. **New `frontend/tailwind.config.ts`**
   v4's CSS-first `@theme` config isn't understood by v3, so every design
   token (colors, radii, font families) that used to live in the
   `@theme inline { ... }` block in `globals.css` now lives here instead,
   as `theme.extend`.

3. **Custom variants ported to a Tailwind plugin**
   The old `@import "shadcn/tailwind.css"` supplied `data-open:`,
   `data-closed:`, `data-active:`, etc. variants via v4's `@custom-variant`
   CSS at-rule (v4-only syntax). These are re-implemented as an
   `addVariant`-based plugin at the top of `tailwind.config.ts`.

4. **`tw-animate-css` → `tailwindcss-animate`**
   `tw-animate-css` is written entirely in v4 CSS-first syntax
   (`@theme`, `@utility`, `--value()`) and does not work under v3 at all —
   under v3's engine those at-rules just pass through as invalid CSS,
   silently breaking every `animate-in` / `fade-in` / `zoom-in` /
   `slide-in-from-*` utility (used ~60+ times across the UI for Radix/Base
   UI enter-exit transitions). `tailwindcss-animate` is the classic v3-era
   plugin providing the same utility names via the JS plugin API.

5. **`babel.config.js` added**
   Forces Next.js off its native SWC compiler and onto Babel (pure JS,
   works on any CPU). This is Next's own documented fallback mechanism —
   the mere presence of a babel config file disables SWC automatically.
   Trade-off: noticeably slower compiles, and a few SWC-only optimizations
   are unavailable.

6. **`globals.css`**: `@import "tailwindcss"` → the v3
   `@tailwind base/components/utilities` directives; the `@theme`/
   `@custom-variant`/v4-only imports removed (ported to config, see above).

## Known visual trade-off — read before assuming something is "broken"

Tailwind v3 cannot apply an **opacity modifier** (the `/50` in classes like
`bg-input/50`, `ring-destructive/20`, `dark:bg-input/30`) to a color that's
defined as a bare `var(--x)` reference — which is how every semantic color
in this project is defined (`--input`, `--ring`, `--destructive`, etc. all
resolve to `oklch(...)`). This was verified directly: under v3.4.19, such a
class silently generates **no CSS rule at all** (not even the solid color —
fully unstyled) rather than erroring.

This affects a real, if modest, number of spots across the shadcn/ui
component set — mostly disabled-state overlays, hover tints, and subtle
focus rings (e.g. `frontend/src/components/ui/textarea.tsx`'s
`disabled:bg-input/50`). One instance that *would* have hard-failed the
build outright (`@apply outline-ring/50` in `globals.css`, inside
`@layer base`) was fixed directly with `color-mix()`, since `@apply` on an
unresolvable utility is a build error, not a silent no-op — the rest are
left as a known, cosmetic-only trade-off.

**Fixing this fully** (full visual parity with the v4 branch) would mean
decomposing every `oklch(...)` value across all six theme blocks
(`:root`/dark, `.dark`, `.light`, `.ocean`, `.forest`, `.sunset`) into
separate L/C/H custom properties and switching every color token to
Tailwind's `<alpha-value>` placeholder pattern — a much larger, riskier
change than "make it boot." Not done here; flag if you want it.

## Verifying elsewhere

This branch was built and smoke-tested with `npm run build` on a machine
*with* AVX2 (to validate the config wiring itself), since a real
crash-reproduction test needs the actual non-AVX2 hardware. If you're
testing on such a machine, `next dev` should no longer crash-loop —
confirm with `lscpu | grep -i avx2` (absent) and a clean `npm run dev` +
page load.
