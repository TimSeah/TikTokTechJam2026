# Phase 0 — Environment & Repo Setup: Completion Record

Original goal: select and verify one GPU execution path within 45 minutes, install the remaining dependencies,
and create the repository skeleton before model implementation expands beyond a 200-image slice.

## Compute-path priority

1. **Native Windows AMD (preferred):** RX 7900 XTX with the official PyTorch 2.9.1 + ROCm 7.2.1
      wheels, Python 3.12, and Adrenalin 26.2.2. AMD explicitly supports Windows 11 build 26200 and
      this GPU. Follow the current [AMD PyTorch on Windows instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
      rather than installing generic PyPI `torch` wheels.
2. **Google Colab CUDA (ready fallback):** open a GPU runtime while the native environment is being
      prepared. Free GPU type, allocation, and runtime duration are not guaranteed, so save outputs at
      stage boundaries.
3. **WSL2 ROCm (last fallback):** use only if native Windows fails and Colab is unavailable or
      terminates. Install Ubuntu 22.04 or 24.04 under WSL2 and follow AMD's ROCm 7.2.1 WSL instructions.
      Store the repo and data in the Linux filesystem, not `/mnt/c`, to avoid small-file I/O overhead.

Stop native environment debugging at 45 minutes. Stop WSL setup after a further 60 minutes and use
CPU or Colab rather than consuming the implementation window.

## Selected environment (verified 1 September 2026)

- Native Windows 11 build 26200; WSL and Colab are not needed for the core run.
- Radeon RX 7900 XTX with Adrenalin 26.6.4 and 24 GB VRAM.
- Python 3.12.13 in `.venv-amd`; PyTorch 2.9.1 + ROCm 7.2.1; torchvision 0.24.1.
- OpenCLIP `ViT-B-32-quickgelu` with the public `openai` checkpoint: 151,277,313 parameters.
- Cached/warmed 256-image FP16 benchmark: 500.0 images/s, finite `(256, 512)` features, and
      1.21 GiB peak allocated VRAM. Actual dataset throughput will also include image decode and
      preprocessing time.
- Forced CPU inference passed for one image in 8.178 seconds.
- `ROCM_SDK_TARGET_FAMILY=custom` is scoped to this workspace to avoid an upstream ROCm launcher
      issue when the virtual-environment path contains spaces.
- The same environment completed the final 9-stage semantic cache, fit, gate, and promotion run in
      547.063 seconds. Every stage passed within the hard 3,600-second budget; timing is recorded in
      [outputs/native_training_timing.json](../../../outputs/native_training_timing.json).

## Steps

1. Confirm the existing GitHub repository is public and the local remote points to it.
2. Start a free Colab GPU runtime and record the assigned GPU using `nvidia-smi`; leave it available
      while testing the native path.
3. Confirm Adrenalin 26.2.2 and 64-bit Python 3.12 are installed. Create a dedicated `.venv-amd`
      environment and install AMD's ROCm SDK and PyTorch wheels exactly as documented.
4. Install non-framework dependencies: `open_clip_torch`, `scikit-learn`, `Pillow`, `numpy`,
      `pandas`, `tqdm`, and `joblib`. Keep the AMD/Colab PyTorch installation separate from generic
      dependency installation so `pip` does not replace the accelerator-specific wheel.
5. Verify the active backend:

      ```bash
      python -c "import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
      ```

6. Run a 256-image CLIP forward pass with `torch.inference_mode()` and FP16 autocast. Record images
      per second and projected time for the planned training and evaluation passes. Accept the path
      only if output features are finite and the projected core run fits within two hours.
7. If native Windows fails any check by minute 45, switch to the Colab GPU. Do not continue native
      debugging in parallel with implementation.
8. Add a `LICENSE` file (MIT or Apache) — the competition rules require custom code to be released
   under one of these.
9. Set up `.gitignore` (`data/`, `*.npy` embedding caches, `outputs/preds.json`, `.venv*/`,
      `__pycache__/`). **Do not** gitignore `outputs/detector.joblib`/model weights or
   `outputs/*.md`/`*.csv` reports — the competition rules require winning teams to open-source
   model weights, so trained artifacts must be committed.
10. Create the target folder skeleton: `src/detector/`, `data/`, `outputs/`.
11. Make all scripts accept `--device auto`; ROCm intentionally uses PyTorch's `torch.cuda` API.

## Definition of done

- [x] One execution path is selected and the unused setup path has been stopped.
- [x] PyTorch reports an available GPU and the expected Radeon or Colab device name.
- [x] The 256-image benchmark produces finite embeddings of the expected shape and a recorded
      throughput estimate.
- [x] A forced CPU run embeds one image successfully, preserving reviewer portability.
- [x] `LICENSE` file present (MIT or Apache).
- [x] Repo skeleton exists locally.
- [ ] Final publication state is verified after the remaining documentation and video work is complete.
- [x] `.gitignore` excludes datasets and large regenerated caches only — trained model weights and
      report files are NOT excluded.

The implementation followed the preferred native Windows AMD path. Reproducible installation now
lives in `requirements-amd.txt`, the generic CUDA/CPU path in `requirements.txt`, and the executable
environment check in `scripts/check_environment.py`. The environment remained stable through the
completed multi-domain retraining and CPU-loadable artifact promotion.

## Time budget

45 minutes for native Windows or Colab selection. WSL2 is allowed one additional 60-minute setup
window only when both preferred paths are unavailable.

## Risks

- Generic `pip install torch` replaces the ROCm wheel — install the accelerator-specific framework
      first and verify `torch.cuda.is_available()` again after installing project dependencies.
- Native Windows package incompatibility — switch at the 45-minute gate; do not troubleshoot
      individual `open_clip`/`torchvision` operators indefinitely.
- Colab runtime termination — write feature caches in restartable shards and copy completed
      artifacts out of `/content` after every phase.
- WSL small-file performance — clone and extract CIFAKE inside the Linux filesystem rather than
      running training against the Windows-mounted checkout.
