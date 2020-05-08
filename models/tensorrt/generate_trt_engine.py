from pathlib import Path
from typing import NoReturn, Optional

import tensorrt as trt

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_engine(
    max_batch_size: int = 1,
    onnx_file_path: Path = "",
    engine_file_path: Optional[Path] = None,
    fp16_mode: bool = False,
    int8_mode: bool = False,
    save_engine: bool = False,
):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(
            TRT_LOGGER
        ) as builder, builder.create_network() as network, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not onnx_file_path.exists():
                quit("ONNX file {} not found".format(str(onnx_file_path)))

            print(f"Loading ONNX file from path {onnx_file_path}...")
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                parser.parse(model.read())

            print("Completed parsing of ONNX file")
            print(
                f"Building an engine from file {onnx_file_path}; this may take a while..."
            )

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if engine_file_path is not None and engine_file_path.exists():
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(str(engine_file_path)))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


if __name__ == "__main__":
    onnx_file = Path(
        "/home/nano/workspace/CenterFace/models/onnx/centerface_1080_1920.onnx"
    )
    engine_file = Path(
        "/home/nano/workspace/CenterFace/models/tensorrt/centerface_fp32_1080_1920.trt"
    )
    get_engine(
        onnx_file_path=onnx_file, engine_file_path=engine_file, save_engine=True,
    )
    print("Done.")
