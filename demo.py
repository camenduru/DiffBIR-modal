import modal, os, sys, shlex

stub = modal.Stub("DiffBIR")
volume = modal.NetworkFileSystem.new().persisted("DiffBIR")

@stub.function(
    image=modal.Image.from_registry("nvidia/cuda:12.2.0-base-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt update -y && \
        apt install -y software-properties-common && \
        apt update -y && \
        add-apt-repository -y ppa:git-core/ppa && \
        apt update -y && \
        apt install -y git git-lfs && \
        git --version  && \
        apt install -y aria2 libgl1 libglib2.0-0 wget && \
        pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
        pip install -q xformers==0.0.20 triton==2.0.0 && \
        pip install -q einops pytorch_lightning gradio omegaconf transformers lpips opencv-python && \
        pip install -q git+https://github.com/mlfoundations/open_clip@v2.20.0"
    ),
    network_file_systems={"/content": volume},
    gpu="A10G",
    timeout=60000,
)
async def run():
    os.environ['HF_HOME'] = '/content/cache/huggingface'
    os.system(f"git clone -b openxlab https://github.com/camenduru/DiffBIR /content/DiffBIR")
    os.chdir(f"/content/DiffBIR")
    os.system(f"git pull")
    os.system(f"git reset --hard")
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/general_full_v1.ckpt -d /content/DiffBIR/models -o general_full_v1.ckpt")
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/general_swinir_v1.ckpt -d /content/DiffBIR/models -o general_swinir_v1.ckpt")
    os.system(f"python gradio_diffbir.py --ckpt /content/DiffBIR/models/general_full_v1.ckpt --config /content/DiffBIR/configs/model/cldm.yaml --reload_swinir --swinir_ckpt /content/DiffBIR/models/general_swinir_v1.ckpt")

@stub.local_entrypoint()
def main():
    run.remote()