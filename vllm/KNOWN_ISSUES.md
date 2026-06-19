
# 01. System Hang During Ubuntu 25.04 Installation with B60 Card Plugged In
The issue is caused by an outdated GPU GuC firmware bundled in the official Ubuntu 25.04 Desktop ISO image.

Workaround: Remove the B60 card before starting the Ubuntu installation, and plug it back in once the installation is complete.
We are also working with the Ubuntu team to address this issue upstream.

# 02. Limited 33 GB/s Bi-Directional P2P Bandwidth with 1x GPU Card
When using a single GPU card over a x16 PCIe connection without a PCIe switch, the observed bi-directional P2P bandwidth is limited to 33 GB/s.

Workaround: Change the PCIe slot configuration in BIOS from Auto/x16 to x8/x8.
With this change, over 40 GB/s bi-directional P2P bandwidth can be achieved.
Root cause analysis is still in progress.

# 03. Container OOM killed (and vllm performance drop) when starting container not by /bin/bash and not run `source /opt/intel/oneapi/setvars.sh`

When using `--enable-auto-tool-choice` and deploy container by docker-compose without `source /opt/intel/oneapi/setvars.sh`, the LD_LIBRARY_PATH will be different and cause the container OOM (or performance drop). It can be reproduced by this two command:

```bash
docker run --rm  --entrypoint "/bin/bash" --name=test intel/llm-scaler-vllm:latest -c env | grep LD_LIBRARY_PATH
 
docker run --rm --entrypoint "/bin/bash" --name=test intel/llm-scaler-vllm:latest -c "source /opt/intel/oneapi/setvars.sh --force && env | grep LD_LIBRARY_PATH"
```

So we need to run `source /opt/intel/oneapi/setvars.sh --force` to ensure some configurations are consistent.

# 04. Multi-GPU device discovery fails on dual+ Arc B-series (Battlemage): `UR_RESULT_ERROR_UNKNOWN` / `torch.xpu.device_count()` throws

On hosts with **more than one** Battlemage GPU (e.g. 2x Intel Arc Pro B70, device `0xe223`), enabling both GPUs in one process fails during device enumeration:

```text
RuntimeError: level_zero backend failed with error: 2147483646 (UR_RESULT_ERROR_UNKNOWN)
```

This blocks `torch.xpu.device_count()`, multi-device `sycl::context(...)`, XCCL all-reduce, and vLLM tensor-parallel serving (`-tp` > 1). **Single-GPU works fine.** See intel/llm-scaler#463, and the upstream root-cause threads intel/compute-runtime#921 and #916.

### Root cause
Starting with the oneAPI 2025.3 / `torch 2.10+` stack, the **Level Zero V2 UR adapter is the default on Xe2 (Battlemage) GPUs**. During multi-device `urContextCreate`, the V2 adapter performs an eager peer-residency handshake (`zeContextMakeMemoryResident` onto each peer device) that the legacy V1 adapter never did. On Battlemage that handshake calls into compute-runtime's peer-import path, which the `xe` kernel driver rejects (peer-imported DMA-buf bind returns `EINVAL` on pre‑7.1 kernels). compute-runtime maps that to `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, and the V2 adapter surfaces it as `UR_RESULT_ERROR_UNKNOWN`.

### Fix
The fix has two parts:

1. **GPU user-mode driver (in this image):** upgrade compute-runtime / Level-Zero to **>= 26.22.38646.4**. The default image now does this (see `docker/Dockerfile`, `NEO_VERSION` build arg). This carries:
   - `e91243a9` — fixes device discovery / multi-device context, **even on the pre‑7.1 host kernels** shipped by this stack.
   - `258518f8` — fixes cross-device P2P data corruption / false OOM (requires a host kernel >= 7.1 to take effect).
2. **Host kernel (outside Docker — containers share the host kernel):** for correct, full-speed cross-device P2P (XCCL collectives, vLLM `-tp>1`), the **host must run kernel >= 7.1** (mainline 7.1 carries the `xe` dma-buf `MOVE_NOTIFY` / peer-bind fixes). Note the bundled `tools/platform/installation/install_kernel.sh` currently pins `6.14.0-1006-intel`; that must be upgraded to a 7.1+ kernel for the P2P fix to apply.

### Stopgap on pre‑7.1 kernels (no rebuild required)
If you cannot yet move the host to kernel >= 7.1, the following environment variables restore dual-GPU device discovery and TP serving by disabling the V2 adapter path and routing collectives through host memory (lower performance, but functional):

```bash
# docker run ... \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0 \
  -e NEOReadDebugKeys=1 \
  -e RenderCompressedBuffersEnabled=0 \
  -e CCL_ENABLE_SYCL_KERNELS=0 \
  -e CCL_ALLREDUCE=direct
```

Or, in docker-compose:

```yaml
    environment:
      SYCL_UR_USE_LEVEL_ZERO_V2: "0"
      SYCL_PI_LEVEL_ZERO_USM_RESIDENT: "0"
      NEOReadDebugKeys: "1"
      RenderCompressedBuffersEnabled: "0"
      CCL_ENABLE_SYCL_KERNELS: "0"
      CCL_ALLREDUCE: "direct"
```

Once the host is on kernel >= 7.1 **and** the image carries compute-runtime >= 26.22.38646.4, these workaround variables should be removed to regain full performance and the V2 code path.
