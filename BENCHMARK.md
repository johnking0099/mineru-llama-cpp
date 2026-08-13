# Two-Step Extraction Benchmarks

MinerU2.5-Pro-2605-1.2B-Q8_0 (Q8_0 quantized, ~506 MB model + ~677 MB mmproj)  
Test image: 660 KB PNG with text, tables, title, page_number  
Method: `MinerUClient(backend="llama-cpp-engine").two_step_extract(image)`  
Config: `n_ctx_seq=8192, n_parallel=4`

## Results (sorted by inference speed)

| # | OS / Arch | CPU | GPU / Backend | Rounds × Iters | Mean Time | Min | Max |
|---|---|---|---|---|---|---|
| 1 | Linux x86_64 | Intel Core i7-8700 | NVIDIA RTX 2080 (8 GB) / Vulkan | 3 × 6 | **6.55s** | — | — |
| 2 | Linux x86_64 | AMD Ryzen AI Max+ 395 | AMD Radeon 8060S / Vulkan | 3 × 6 | **7.69s** | — | — |
| 3 | macOS 14 arm64 | Apple M1 Max (10 cores) | Metal | 3 × 6 | **10.94s** | — | — |
| 4 | Linux x86_64 | Intel Xeon Platinum 8358P | CPU (AVX-512, musl, docker) | 3 × 6 | **25.51s** | 24.00s | 26.98s |
| 5 | Linux x86_64 | Intel Xeon Platinum 8358P | CPU (AVX-512, glibc, docker) | 3 × 6 | **25.67s** | 24.27s | 26.86s |
| 6 | Linux x86_64 | Intel Xeon Platinum 8358P | CPU (AVX-512, native) | 3 × 6 | **29.90s** | — | — |
| 7 | macOS 14 arm64 | Apple M1 Max (10 cores) | CPU | 1 × 6 | **80.02s** | 76.81s | 82.60s |
| 8 | Linux aarch64 | Apple M1 Max (docker) | CPU (glibc) | 1 × 6 | **126.14s** | 123.86s | 128.46s |
| 9 | Linux aarch64 | Apple M1 Max (docker) | CPU (musl) | 1 × 6 | **130.31s** | 125.39s | 132.70s |
| 10 | Windows 11 ARM64 | Apple Silicon (VMware 2 vCPU) | CPU | 1 × 6 | **304.92s** | 302.86s | 310.07s |
| 11 | Windows 10 x86_64 | Intel Core i5-10210U | Intel UHD Graphics / Vulkan | 3 × 6 | **479.78s** | 441.78s | 520.25s |

## Notes

- Rounds × Iters = 3 × 6 means 18 total zwei-step extractions per platform; 1 × 6 = 6 total.
- "Vulkan" backend is enabled via `n_gpu_layers=99`; "CPU" via `n_gpu_layers=0`.
- macOS arm64 Metal is ~7.3× faster than CPU on the same machine.
- M1 Max performance varies significantly by runtime: native Metal (10.9s) vs docker Linux ARM CPU (126s) vs native macOS CPU (80s).
- Docker on macOS ARM incurs a lightweight Linux VM overhead (~57% slower than native macOS CPU on identical hardware).
- Intel UHD Graphics (Comet Lake-U, 24 EU) Vulkan is slower than CPU on the same machine (478s vs previously measured ~241s CPU-only).
- All platforms produced identical 7-block detection results (consistent correctness).