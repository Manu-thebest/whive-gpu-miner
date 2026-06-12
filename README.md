# Whive GPU Miner (Yespower)

**🤖 AI-Generated Proof of Concept**

> This project was autonomously developed by **Cortana** (an AI agent running on [Hermes Agent](https://hermes-agent.nousresearch.com)) through iterative debugging and testing on a real AMD RX 580 GPU. It is a **proof of concept** demonstrating hybrid GPU acceleration for Yespower — not intended for production mining.

Hybrid CPU+GPU miner for **Whive** (Yespower algorithm — YESPOWER_1.0, N=2048, r=32).

**Architecture:** The GPU handles the memory-intensive `smix` phase (S-init + blockmix + pwxform), while the CPU manages SHA256, PBKDF2, HMAC, and Stratum networking. This design avoids GPU SHA256 bugs common on AMD hardware.

## Features

- ✅ Hybrid CPU+GPU mining (AMD RX 580 tested, 8GB)
- ✅ Stratum protocol (subscribe, authorize, job notify, submit)
- ✅ Pool difficulty tracking
- ✅ Real-time stats (hashrate, shares)
- ✅ OpenCL 3.0 with `cl_khr_int64`
- ✅ Verifiable against `yespower_tls` reference

## Requirements

- **GPU:** AMD RX 580 (36 CUs) or similar, 8GB VRAM
- **OpenCL:** 3.0 with `cl_khr_int64` (rusticl on Mesa)
- **CPU:** Any x86_64 with yespower-ref.o
- **Deps:** `gcc`, `libOpenCL`, `pthreads`

## Build

```bash
git clone https://github.com/Manu-thebest/whive-gpu-miner
cd whive-gpu-miner
git submodule update --init --recursive  # if using submodules

# Compile reference library
cd whive-cpuminer-ref/yespower && make && cd ../..

# Build miner
gcc -O3 -o gpu_miner gpu_miner.c \
    -Iwhive-cpuminer-ref/yespower \
    -lOpenCL -lm \
    whive-cpuminer-ref/yespower/yespower-ref.o \
    whive-cpuminer-ref/yespower/sha256.o \
    -lpthread
```

## Usage

Edit `gpu_miner.c` to set your pool, wallet, and worker:

```c
#define POOL_HOST   "your.pool.com"
#define POOL_PORT   3333
#define WALLET      "YOUR_WALLET_ADDRESS"
#define WORKER      "YOUR_WORKER_NAME"
```

Then run:

```bash
./gpu_miner
```

### Output

```
[12:34:56] === Whive GPU Miner ===
[12:34:56] Pool: 206.189.2.17:3333 Wallet: WRKn5ox9.../WORKER
[12:34:56] GPU: AMD Radeon RX 580 Series
[12:34:56] Kernel built
[12:34:56] Connected
[12:34:56] Subscribe: en1=07ffcedf
[12:34:56] Difficulty: 4
[12:34:56] Authorize: OK
[12:35:00] Job b0c5: ver=20000000 bits=1e016473
[12:35:24] Stats: 9 H/s total=640 acc=1 rej=0
```

## How It Works

1. **Stratum** connects to pool, subscribes, authorizes
2. **Job loop:** receives mining.notify, builds block headers
3. **GPU batch:** prepares 64 headers, runs OpenCL kernel for smix
4. **CPU finalize:** HMAC-SHA256 each candidate, check target
5. **Submit:** valid shares sent back via stratum

The GPU kernel (`yespower_smix.cl`) runs the memory-hard Salsa20/8 blockmix + pwxform that makes yespower ASIC-resistant.

## Performance

| Setup | Hashrate | Notes |
|-------|----------|-------|
| RX 580 (batch=64) | ~10 H/s | 8MB/hash, 2GB VRAM |
| cpuminer-opt (CPU) | ~530 H/s | All cores, reference impl |

> Note: Yespower is memory-hard by design — GPU hashrate is lower than CPU on this algorithm due to random 8MB memory access per hash. The GPU miner is a proof-of-concept for hybrid GPU acceleration.

## Files

| File | Description |
|------|-------------|
| `gpu_miner.c` | Main miner (Stratum + CPU logic + GPU orchestration) |
| `yespower_smix.cl` | OpenCL kernel (S-init + blockmix + pwxform) |
| `test_hybrid_full.c` | Verifies GPU kernel matches CPU reference |
| `opencl_test_yespower.c` | Debug/test utility |

## License

MIT
