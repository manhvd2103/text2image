import argparse
import os
import math
import shutil
import onnx
import torch
import onnxscript

from pathlib import Path
from packaging import version
from torch.onnx import export
from onnxscript.onnx_opset import opset17 as op

from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")


custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=1)


@onnxscript.script(custom_opset)
def ScaledDotProductAttention(
    query,
    key,
    value,
    dropout_p,
):
    # Swap the last two axes of key
    key_shape = op.Shape(key)
    key_last_dim = key_shape[-1:]
    key_second_last_dim = key_shape[-2:-1]
    key_first_dims = key_shape[:-2]

    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    embedding_size = op.CastLike(op.Shape(query)[-1], query)
    scale = op.Div(1.0, op.Sqrt(embedding_size))

    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    attn_weight = op.Softmax(
        op.MatMul(query_scaled, key_transposed_scaled),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


def custom_scaled_dot_product_attention(g, query, key, value, attn_mask, dropout, is_causal, scale=None):
    return g.onnxscript_op(ScaledDotProductAttention, query, key, value, dropout).setType(value.type())


torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::scaled_dot_product_attention",
    symbolic_fn=custom_scaled_dot_product_attention,
    opset_version=17,
)

def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False
):
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset
        )

@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int, fp16: bool=False):
    torch.onnx.register_custom_op_symbolic
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ("float 16 model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"
    
    pipeline = StableDiffusionPipeline.from_single_file("weights/stable_diffusion/model.safetensors", torch_dtype=dtype, safetensors=True).to(device)
    output_path = Path(output_path)

    # Text encoder
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    onnx_export(
        pipeline.text_encoder,
        model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
        output_path=output_path / "text_encoder" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        opset=opset
    )

    del pipeline.text_encoder

    # UNET
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.rand(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            torch.rand(2).to(device=device, dtype=dtype),
            torch.rand(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            False
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"}
        },
        opset=opset,
        use_external_data_format=True
    )
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)
    onnx.save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False
    )
    del pipeline.unet

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)[0].sample()
    onnx_export(
        vae_encoder,
        model_args=(
            torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_encoder" / "model.onnx",
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )
    del pipeline.vae

    # SAFETY CHECKER
    if pipeline.safety_checker is not None:
        safety_checker = pipeline.safety_checker
        clip_num_channels = safety_checker.config.vision_config.num_channels
        clip_image_size = safety_checker.config.vision_config.image_size
        safety_checker.forward = safety_checker.forward_onnx
        onnx_export(
            pipeline.safety_checker,
            model_args=(
                torch.randn(
                    1,
                    clip_num_channels,
                    clip_image_size,
                    clip_image_size,
                ).to(device=device, dtype=dtype),
                torch.randn(1, vae_sample_size, vae_sample_size, vae_out_channels).to(device=device, dtype=dtype),
            ),
            output_path=output_path / "safety_checker" / "model.onnx",
            ordered_input_names=["clip_input", "images"],
            output_names=["out_images", "has_nsfw_concepts"],
            dynamic_axes={
                "clip_input": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "images": {0: "batch", 1: "height", 2: "width", 3: "channels"},
            },
            opset=opset,
        )
        del pipeline.safety_checker
        safety_checker = OnnxRuntimeModel.from_pretrained(output_path / "safety_checker")
        feature_extractor = pipeline.feature_extractor
    else:
        safety_checker = None
        feature_extractor = None

    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
        vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
        text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
        scheduler=pipeline.scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        requires_safety_checker=safety_checker is not None,
    ).to("cuda")

    onnx_pipeline.save_pretrained(output_path)
    print("ONNX pipeline saved to", output_path)

    del pipeline
    del onnx_pipeline
    _ = OnnxStableDiffusionPipeline.from_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
        default="weights/stable_diffusion/model_v3.safetensors"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="weights/onnx",
        help="Path to the output model."
    )
    parser.add_argument(
        "--opset",
        default=17,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Export the models in `float16` mode"
    )
    args = parser.parse_args()
    convert_models(args.model_path, args.output_path, args.opset, args.fp16)
