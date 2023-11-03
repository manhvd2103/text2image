import os
import re
import string
import json
import torch
import random
import time
import shutil
import torch
import random
import logging

from torch import autocast
from pynvml import *
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler
)
from inference_realesrgan import esrgan_infer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def txt2img(
    prompt:str="best quality, 1picture chill music",
    negative_prompt:str="",
    stable_diffusion_weights:str="weights/stable_diffusion/model_v2.safetensors",
    save_path:str="static/images/inputs",
    lora_weight:str=None,
    output_path:str="/home/www/data/data/saigonmusic/hg_project_effect/",
    height:int=512,
    width:int=768,
    name:str=None,
    guidance_scale=7.5,
    num_images_per_prompt=1,
    num_inference_steps:int=80,
    seed=2937362614,
    lora_scale=0.0,
    suffix:str=""
):
    # Define the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Normalize input prompt and negative prompt
    prompt = prompt.lower()
    negative_prompt = negative_prompt.lower()

    # Set the height and width (`height` and `width` must be divisible by 8)
    if height % 8 != 0:
        height = height - height % 8
    if width % 8 != 0:
        width = width - width % 8

    # Load model with key words
    if stable_diffusion_weights.split(".")[-1] == "safetensors":

        # Key word [relax]
        if prompt == "[relax]":
            logger.info("2D mode for relax model")

            # Load random prompt and negative_prompt with key word [relax]
            with open("list_prompts/relax.json", "r", encoding="utf-8") as f:
                relax_data = json.load(f)
                random_key = random.choice(list(relax_data.keys()))
                random_relax_data = relax_data[f"{random_key}"]

            # Load stable diffusion, lora and vae weight and define the lora
            # scale
            stable_diffusion_weights = os.path.join(
                "weights/stable_diffusion",
                random_relax_data["model"] + ".safetensors"
            )
            lora_weight = os.path.join(
                "weights/Lora/",
                random_relax_data["lora"] + ".safetensors"
            )

            # Get prompt and negative_prompt
            prompt = random_relax_data["prompt"]
            negative_prompt = random_relax_data["negative"]

            # Get lora scale
            pattern = "\d*\.\d+|\d+"
            lora_scale = float(re.findall(pattern, prompt)[0])
            lora_scale = lora_scale if lora_scale <= 1.0 else 1.0
        elif prompt == "[sea]":
            logger.info("2D model for sea style model")
            with open("list_prompts/sea.json", "r", encoding="utf-8") as f:
                # Load random data with key word [sea]
                sea_data = json.load(f)
                random_key = random.choice(list(sea_data.keys()))
                random_sea_data = sea_data[f"{random_key}"]

                # Weight for stable_diffusion model
                stable_diffusion_weights = "weights/stable_diffusion/model_v7.safetensors"

                # Get random prompt and negative prompt
                prompt = random_sea_data["prompt"]
                negative_prompt = random_sea_data["negative"]

        # Define the stable diffusion pipeline
        logger.info(f"Load weight from {stable_diffusion_weights}")
        pipeline = StableDiffusionPipeline.from_single_file(
            stable_diffusion_weights,
            safetensors=True,
            torch_type=torch.float16
        )
        pipeline.scheduler = DPMSolverSDEScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
        if lora_weight is not None:
            logger.info(f"Load lora weight from {lora_weight}")
            pipeline.load_lora_weights(lora_weight)
        pipeline.to(device)
    else:
        raise NotImplementedError("This weights is not supported! Please use safetensors weight")

    # Generate images
    with autocast("cuda"), torch.no_grad():
        logger.info("Start generate image:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        logger.info(f"Lora scale: {lora_scale}")
        tic = time.time()

        # Get max length of clip tokenizer
        max_length = pipeline.tokenizer.model_max_length

        # Get number token of prompt and negative_prompt
        count_prompt = len(prompt.split(" "))
        count_negative_prompt = len(negative_prompt.split(" "))

        # Create the tensor based on which prompt is longer
        if count_prompt >= count_negative_prompt:
            input_ids = pipeline.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False
            ).input_ids.to(device)
            shape_max_length = input_ids.shape[-1]
            negative_ids = pipeline.tokenizer(
                negative_prompt,
                truncation=False,
                padding="max_length",
                max_length=shape_max_length,
                return_tensors="pt"
            ).input_ids.to(device)
        else:
            negative_ids = pipeline.tokenizer(
                negative_prompt,
                truncation=False,
                return_tensors="pt"
            ).input_ids.to(device)
            shape_max_length = negative_ids.shape[-1]
            input_ids = pipeline.tokenizer(
                prompt,
                truncation=False,
                padding="max_length",
                return_tensors="pt",
                max_length=shape_max_length
            ).input_ids.to(device)

        # Get prompt_embeds and negative_prompt_embeds
        concat_embeds = []
        neg_embeds = []

        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(
                pipeline.text_encoder(input_ids[:, i: i + max_length])[0]
            )
            neg_embeds.append(
                pipeline.text_encoder(negative_ids[:, i: i + max_length])[0]
            )
        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        # Generate image with concated prompt and negative prompt
        images = pipeline(
            prompt_embeds=prompt_embeds,
            num_inference_steps=num_inference_steps,
            height=height, width=width,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            cross_attention_kwargs={"scale": lora_scale}
        ).images[0]

        # Save image and up scale if resolution of image is low
        if name is None:
            name = "".join(random.choice(string.ascii_letters) for i in range(10)) + ".jpg"
        path = os.path.join(save_path, name)
        display_path = os.path.join('static/images/outputs', name)
        images.save(path)
        print(f"Generative done! The image save in {path}")
        if height < 1080 and width < 1920:
            esrgan_model_name = "RealESRGAN_x4plus_anime_68"
            esrgan_infer(
                input_img=path,
                model_name=esrgan_model_name,
                output_img=output_path,
                suffix=suffix
            )
        else:
            shutil.copyfile(path, os.path.join(output_path, name))
            print(f"Remove image with lower resolution!")
            os.remove(path)
            shutil.copyfile(os.path.join(output_path, name), display_path)
        toc = time.time()
        print(f"Everything done in {round(toc - tic, 2)} seconds")

    # Delete model and clear CUDA cache
    del pipeline
    torch.cuda.empty_cache()
    return display_path

if __name__ == "__main__":

    while True:
        try:
            wait_list = os.listdir("request_prompts")
            for wl in wait_list:
                nvmlInit()
                h = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(h)
                if info.used / info.total < 0.4:
                    json_f = open(
                        os.path.join("request_prompts", wl),
                        "r",
                        encoding="utf-8"
                    )
                    data = json.load(json_f)
                    json_f.close()
                    prompts = data["prompts"]
                    name = data["name"]
                    path = txt2img(prompt=prompts, name=name)
                    os.remove(os.path.join("request_prompts", wl))
                else:
                    continue
        except Exception as e:
            print(e)
