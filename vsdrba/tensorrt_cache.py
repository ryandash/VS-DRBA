"""Raw TensorRT engine serialization helpers for Torch-TensorRT-RTX."""

from __future__ import annotations

import gc
import os
import uuid


def _runtime_module(engine_bytes, inputs, label):
    """Wrap raw engine bytes in Torch-TensorRT's lightweight Python runtime."""
    import torch_tensorrt

    device = inputs[0].device
    # Runtime settings describe the target device. Precision is already fixed
    # in the serialized engine and is not a CompilationSettings runtime option
    # in the installed TensorRT-RTX 2.11 API.
    settings = torch_tensorrt.dynamo.CompilationSettings(
        device=torch_tensorrt.Device(
            gpu_id=device.index if device.index is not None else torch.cuda.current_device()
        )
    )
    # The runtime module needs binding names to connect its PyTorch tensor
    # arguments to TensorRT I/O tensors. Discover them from the raw plan.
    import tensorrt_rtx as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize the engine for {label}")
    input_names = []
    output_names = []
    for index in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_names.append(tensor_name)
        else:
            output_names.append(tensor_name)
    del engine, runtime

    # Torch-TensorRT releases expose different names for this runtime wrapper.
    # The standard package uses PythonTorchTensorRTModule; the RTX build in
    # some Windows distributions exposes TorchTensorRTModule instead.
    runtime_namespace = torch_tensorrt.runtime
    runtime_class = getattr(runtime_namespace, "PythonTorchTensorRTModule", None)
    if runtime_class is None:
        runtime_class = getattr(runtime_namespace, "TorchTensorRTModule", None)
    if runtime_class is None:
        available = [
            name for name in dir(runtime_namespace)
            if "TensorRTModule" in name
        ]
        raise RuntimeError(
            "This Torch-TensorRT installation exposes no supported TensorRT "
            f"runtime module. Available candidates: {available}"
        )

    runtime_module = runtime_class(
        serialized_engine=engine_bytes,
        input_binding_names=input_names,
        output_binding_names=output_names,
        name=label,
        settings=settings,
    )
    # Once the runtime has created its engine object, release the duplicate
    # serialized host buffer if this implementation exposes it. Do not clear
    # it before engine initialization, since lazy runtime variants need it.
    if getattr(runtime_module, "engine", None) is not None and hasattr(
        runtime_module, "serialized_engine"
    ):
        runtime_module.serialized_engine = None
    print(f"Loaded raw TensorRT engine: {label} on {device}")
    return runtime_module


def load_or_compile(
    module,
    inputs,
    artifact_path: str,
    compile_kwargs: dict,
    label: str,
    dynamic_shapes=None,
):
    """Load a standalone .engine file or build and serialize one.

    The artifact contains raw TensorRT engine bytes, not a PyTorch ExportedProgram.
    PyTorch tensors are still used at the VapourSynth integration boundary; the
    runtime module only wraps TensorRT execution and does not retain the source
    neural-network module or its ExportedProgram.
    """
    import torch
    import torch_tensorrt

    artifact_path = os.path.realpath(os.path.expanduser(artifact_path))
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    if os.path.isfile(artifact_path) and os.path.getsize(artifact_path) > 0:
        try:
            with open(artifact_path, "rb") as engine_file:
                engine_bytes = engine_file.read()
            runtime_module = _runtime_module(engine_bytes, inputs, label)
            del engine_bytes
            gc.collect()
            return runtime_module
        except Exception as exc:
            raise RuntimeError(
                f"Could not load serialized TensorRT engine for {label} at "
                f"{artifact_path}. Remove the incompatible artifact to rebuild it: {exc}"
            ) from exc

    print(f"Building standalone TensorRT engine for {label}; this is a one-time build.")
    exported_program = torch.export.export(
        module, tuple(inputs), dynamic_shapes=dynamic_shapes
    )

    # This API returns the actual serialized ICudaEngine bytes. Unlike
    # dynamo.compile(), it does not produce a PyTorch GraphModule or an
    # ExportedProgram wrapper. Full compilation is required so no PyTorch
    # fallback subgraphs are silently omitted from the standalone engine.
    converter_kwargs = {
        "inputs": list(inputs),
        "enabled_precisions": {inputs[0].dtype},
        "device": compile_kwargs.get("device", inputs[0].device),
        "num_avg_timing_iters": compile_kwargs.get("num_avg_timing_iters", 1),
        "workspace_size": compile_kwargs.get("workspace_size", 0),
        "min_block_size": compile_kwargs.get("min_block_size", 1),
        "require_full_compilation": True,
        "pass_through_build_failures": compile_kwargs.get(
            "pass_through_build_failures", True
        ),
        "max_aux_streams": compile_kwargs.get("max_aux_streams"),
        "optimization_level": compile_kwargs.get("optimization_level"),
        "use_explicit_typing": compile_kwargs.get("use_explicit_typing", True),
        "runtime_cache_path": compile_kwargs.get("runtime_cache_path"),
    }
    engine_bytes = (
        torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported_program, **converter_kwargs
        )
    )
    del exported_program
    gc.collect()

    if not isinstance(engine_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(
            "Torch-TensorRT raw engine converter did not return serialized bytes; "
            f"got {type(engine_bytes).__name__}"
        )

    temporary_path = artifact_path + f".{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "wb") as engine_file:
            engine_file.write(engine_bytes)
            engine_file.flush()
            os.fsync(engine_file.fileno())
        os.replace(temporary_path, artifact_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    runtime_module = _runtime_module(engine_bytes, inputs, label)
    del engine_bytes
    gc.collect()
    return runtime_module
