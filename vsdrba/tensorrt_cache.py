"""Persistent AOT serialization helpers for Torch-TensorRT-RTX modules."""

from __future__ import annotations

import os
import uuid


def load_or_compile(
    module,
    inputs,
    artifact_path: str,
    compile_kwargs: dict,
    label: str,
    dynamic_shapes=None,
):
    """Load a serialized TensorRT module, or compile and atomically save it.

    Artifacts use TorchScript so they can be loaded in a fresh Python process.
    Runtime JIT cache paths remain separate and are passed through compile_kwargs.
    """
    import torch
    import torch_tensorrt

    artifact_path = os.path.realpath(os.path.expanduser(artifact_path))
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    if os.path.isfile(artifact_path) and os.path.getsize(artifact_path) > 0:
        try:
            loaded = torch_tensorrt.load(artifact_path).module().eval()
            print(f"Loaded serialized TensorRT engine: {label} ({artifact_path})")
            return loaded
        except Exception as exc:
            raise RuntimeError(
                f"Could not load serialized TensorRT engine for {label} at "
                f"{artifact_path}. Remove the incompatible artifact to rebuild it: {exc}"
            ) from exc

    print(f"Building TensorRT engine for {label}; this is a one-time build for this cache key.")
    exported_program = torch.export.export(
        module, tuple(inputs), dynamic_shapes=dynamic_shapes
    )
    compiled = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=list(inputs),
        **compile_kwargs,
    ).eval()

    # Save to a unique sibling path, then publish atomically. A process interrupted
    # during serialization cannot leave a truncated artifact at the final path.
    temporary_path = artifact_path + f".{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        torch_tensorrt.save(
            compiled,
            temporary_path,
            arg_inputs=list(inputs),
            output_format="torchscript",
        )
        os.replace(temporary_path, artifact_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return compiled
