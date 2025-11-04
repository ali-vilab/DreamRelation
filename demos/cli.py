#! /usr/bin/env python
import json
import os
import time
import sys

import click
import numpy as np
import torch

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

import yaml

def parse_comma_separated(ctx, param, value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(',')]
    return value

pipeline = None
model_dir_path = None
lora_path = None
num_gpus = torch.cuda.device_count()
cpu_offload = False
attention_mode = None
train_cfg = None
test_lora_names = None


def configure_model(model_dir_path_, lora_path_, cpu_offload_, attention_mode_=None, train_cfg_path_=None, test_lora_names_=None):
    global model_dir_path, lora_path, cpu_offload, load_custom_lora, attention_mode, train_cfg, test_lora_names
    model_dir_path = model_dir_path_
    lora_path = lora_path_
    cpu_offload = cpu_offload_
    attention_mode = attention_mode_
    test_lora_names = test_lora_names_
    if train_cfg_path_ is not None:
        with open(train_cfg_path_, 'r', encoding='utf-8') as train_file:
            train_cfg = yaml.safe_load(train_file)

def load_model():
    global num_gpus, pipeline, model_dir_path, lora_path, attention_mode
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(
                model_path=f"{MOCHI_DIR}/dit.safetensors",
                lora_path=lora_path,
                model_dtype="bf16",
                attention_mode=attention_mode,
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/decoder.safetensors",
            ),
        )
        if num_gpus > 1:
            assert not lora_path, f"Lora not supported in multi-GPU mode"
            assert not cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = num_gpus
        else:
            kwargs["cpu_offload"] = cpu_offload
            kwargs["decode_type"] = "tiled_spatial"
            # kwargs["decode_type"] = "tiled_full"
            kwargs["fast_init"] = not lora_path
            kwargs["strict_load"] = not lora_path
            kwargs["decode_args"] = dict(overlap=8)

        #! model kwargs
        if train_cfg is not None:
            model_kwargs = train_cfg.get('model', {}).get('kwargs', {})
            kwargs["model_kwargs"] = model_kwargs
            if test_lora_names is not None:
                kwargs["model_kwargs"]["test_lora_names"] = test_lora_names

        pipeline = klass(**kwargs)

def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
):
    output_dir = f"outputs/{prompt.replace(' ', '_').replace('.', '')}"
    # test_lora_names = []

    assert lora_path is not None, "Please provide a valid lora path"
    
    train_step = lora_path.rsplit('/', 1)[-1].split('.')[0].split('_')[1]
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"lora_{train_step}_{'_'.join(test_lora_names)}_seed{str(seed)}.mp4"
    output_path = os.path.join(output_dir, base_filename)

    if os.path.exists(output_path):
        print(f"------------------------ Output file already exists: {output_path}")
        return output_path

    load_model()

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_inference_steps,
        # We *need* flash attention to batch cfg
        # and it's only worth doing in a high-memory regime (assume multiple GPUs)
        "batch_cfg": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }

    with progress_bar(type="tqdm"):
        final_frames = pipeline(**args)

        final_frames = final_frames[0]

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        save_video(final_frames, output_path)

        return output_path


# from textwrap import dedent

# DEFAULT_PROMPT = dedent("""
# A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl 
# filled with lemons and sprigs of mint against a peach-colored background. 
# The hand gently tosses the lemon up and catches it, showcasing its smooth texture. 
# A beige string bag sits beside the bowl, adding a rustic touch to the scene. 
# Additional lemons, one halved, are scattered around the base of the bowl. 
# The even lighting enhances the vibrant colors and creates a fresh, 
# inviting atmosphere.
# """)


@click.command()
@click.option("--prompt", required=True, help="Prompt for video generation.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=61, type=int, help="Number of frames.")
@click.option("--seed", default=0, type=int, help="Random seed.")
@click.option("--cfg_scale", default=6.0, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--lora_path", required=False, help="Path to the lora file.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
@click.option("--attention_mode", required=False, help="Attention mode")
@click.option("--train_cfg_path", required=False, help="Path to the training file.")
@click.option("--test_lora_names", required=False, type=str, callback=parse_comma_separated, help="Test lora type, such as relation,subject1,subject2")
def generate_cli(
    prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, model_dir, lora_path, cpu_offload, attention_mode, train_cfg_path, test_lora_names
):
    prompts_list = [prompt] if prompt else []
    
    for current_prompt in prompts_list:
        configure_model(model_dir, lora_path, cpu_offload, attention_mode, train_cfg_path, test_lora_names)
        output = generate_video(
            current_prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            seed,
            cfg_scale,
            num_steps,
        )
        click.echo(f"Video generated at: {output}")

if __name__ == "__main__":
    generate_cli()
