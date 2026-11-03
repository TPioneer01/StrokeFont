import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


def setup_dist():
    """Initialize from torchrun variables, or create a single-process group."""
    if dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        if int(os.environ["WORLD_SIZE"]) > 1:
            raise ValueError("Multi-process execution requires MASTER_PORT; use launch.py or torchrun.")
        os.environ["MASTER_PORT"] = str(_find_free_port())
    if th.cuda.is_available():
        th.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    backend = "nccl" if th.cuda.is_available() and dist.is_nccl_available() else "gloo"
    if os.name == "nt":
        # Windows PyTorch wheels do not include libuv support for TCPStore.
        os.environ.setdefault("USE_LIBUV", "0")

    dist.init_process_group(backend=backend, init_method="env://")


def dev():

    if th.cuda.is_available():
        return th.device("cuda", th.cuda.current_device())
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """Each process reads the checkpoint from a shared accessible path."""
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):

    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
