#!/usr/bin/env bash
# Set up a fresh TPU VM (Ubuntu 22.04, v2-alpha-tpuv6e) for wsp. Expects ~/wsp.tgz (tarred from inside the repo root).
# Verified 2026-11-08 on spot v6e-1, europe-west4-a: torch 2.15.0+cpu, torch_tpu, interp-engine 1.14.3.
set -e
sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
sudo apt-get update -qq && sudo apt-get install -y -qq python3.12 python3.12-venv git >/dev/null
python3.12 -m venv ~/wsp-env && source ~/wsp-env/bin/activate
pip install -q --upgrade pip
pip install -q torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
[ -d ~/interp-engine/.git ] || git clone -q -b feat/tpu-eager https://github.com/TinevimboMusingadi/interp-engine.git ~/interp-engine
pip install -q -e ~/interp-engine --extra-index-url https://download.pytorch.org/whl/cpu
mkdir -p ~/work/Interp-tools-wsp && tar -xzf ~/wsp.tgz -C ~/work/Interp-tools-wsp  # tar made from inside the repo
pip install -q -e "$HOME/work/Interp-tools-wsp[agents]" --extra-index-url https://download.pytorch.org/whl/cpu
python -c "import torch, torch_tpu, interp_engine, wsp, datasets, google.genai; print('accel', torch.accelerator.current_accelerator(), torch.__version__, interp_engine.__version__)"
