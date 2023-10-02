import os
import string
import json
import torch
import random
import time
import shutil
import torch

from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from inference_realesrgan import esrgan_infer


def txt2img(
    prompt:str="best quality, 1picture chill music",
    negative_prompt:str="3d, cartoon, anime, sketches, (worst quality, bad quality, child, cropped:1.4) ((monochrome)), ((grayscale)), (bad-hands-5:1.0), (badhandv4:1.0), (easynegative:0.8), (bad-artist-anime:0.8), (bad-artist:0.8), (bad_prompt:0.8), (bad-picture-chill-75v:0.8), (bad_prompt_version2:0.8), (bad_quality:0.8)",
    stable_diffusion_weights:str="weights/stable_diffusion/model_v2.safetensors",
    save_path:str="static/images/inputs",
    lora_weight:str=None,
    output_path:str="/home/www/data/data/saigonmusic/hg_project_effect/",
    height:int=512,
    width:int=768,
    name:str=None,
    guidance_scale=8,
    num_images_per_prompt=1,
    num_inference_steps:int=80,
    seed=2937362614,
    suffix:str=""
):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    prompt = prompt.lower()
    if height % 8 != 0:
        height = height - height % 8
    if width % 8 != 0:
        width = width - width % 8
    if stable_diffusion_weights.split(".")[-1] == "ckpt":
        print(f"Load weight from {stable_diffusion_weights}")
        pipeline = StableDiffusionPipeline.from_pretrained(stable_diffusion_weights).to(device)
    elif stable_diffusion_weights.split(".")[-1] == "safetensors":
        if "[relax]" in prompt:
            stable_diffusion_weights = "weights/stable_diffusion/model_v5.safetensors"
            lora_weight = "weights/Lora/model_v1.safetensors"
        print(f"Load weight from {stable_diffusion_weights}")
        pipeline = StableDiffusionPipeline.from_single_file(stable_diffusion_weights, safetensors=True)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
        if lora_weight is not None:
            pipeline.load_lora_weights(lora_weight)
        pipeline.to(device)
    else:
        raise NotImplementedError("This weights is not supported! Please use ckpt or safetensors weight")
    with autocast("cuda"), torch.no_grad():
        print(f"Starting generative image with prompt: {prompt} and resolution is {width}x{height}...")
        tic = time.time()
        
        images = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, height=height, width=width, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images
        for image in images:
            if name is None:
                name = "".join(random.choice(string.ascii_letters) for i in range(10)) + ".jpg"
            path = os.path.join(save_path, name)
            display_path = os.path.join('static/images/outputs', name)
            image.save(path)
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

    del pipeline
    torch.cuda.empty_cache()
    return display_path

if __name__ == "__main__":

    while True:
        try:
            wait_list = os.listdir("json_files")
            for wl in wait_list:
                json_f = open(os.path.join("json_files", wl), "r", encoding="utf-8")
                data = json.load(json_f)
                json_f.close()
                prompts = data["prompts"]
                name = data["name"]
                path = txt2img(prompt=prompts, name=name)
                os.remove(os.path.join("json_files", wl))
        except Exception as e:
            print(e)
