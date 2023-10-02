import cv2
import glob
import torch
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.models.utils import RealESRGANer

def esrgan_infer(
    input_img:str="images/inputs",
    model_name:str="RealESRGAN_x4plus",
    output_img:str="images/outputs",
    denoise_streghth:float=0.5,
    out_scale:int=4,
    model_path:str=None,
    tile:str=0,
    suffix:str="out",
    tile_pad:int=10,
    pre_pad:int=0,
    face_enhance:bool=False,
    fp32:bool=False,
    alpha_upsampler:str="realesrgan",
    ext:str="auto",
    gpu_id:int=None
):
    # Determine models according to model names
    model_name = model_name.split(".")[0]
    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]
    if model_name == "RealESRNet_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"]
    if model_name == "RealESRGAN_x4plus_anime_68":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4
        )
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"]
    if model_name == "RealESRGAN_x2plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2
        )
        netscale = 2
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
    if model_name == "realesr-animevideov3":
        model = SRVGGNetCompact(
            num_in_channel=3,
            num_out_channel=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu"
        )
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"]
    if model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(
            num_in_channel=3,
            num_out_channel=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu"
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        ]
    # else:
    #     raise NotImplementedError(f"The model name {model_name} is not implement")
    
    # Determinemodel paths
    if model_path is not None:
        model_path = model_path
    else:
        model_path = os.path.join("weights/esrgan", model_name + ".pth")
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url=url,
                    model_dir=os.path.join(ROOT_DIR, "weights/esrgan"),
                    progress=True,
                    file_name=None
                )
    # Use dni to control the dinoise strength
    dni_weight = None
    if model_name == "realesr-general-x4v3" and denoise_streghth != 1:
        wdn_model_path = model_path.replace("realesr-general-x4v3", "realesr-general-wdn-x4v3")
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_streghth, 1- denoise_streghth]
    
    # Restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=fp32,
        gpu_id=gpu_id
    )

    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=out_scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler
        )
    os.makedirs(output_img, exist_ok=True)
    
    if os.path.isfile(input_img):
        paths = [input_img]
    else:
        paths = sorted(glob.glob(os.path.join(input_img, "*")))
    try:
        for idx, path in enumerate(paths):
            imgname, extension = os.path.splitext(os.path.basename(path))
            print(f"Up resolution for image: {imgname}, please wait a while")

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = "RGBA"
            else:
                img_mode = None
            try:
                if face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=out_scale)
            except RuntimeError as error:
                print("Error", error)
                print("If encounter CUDA out of memory, try to set tile with a smaller number")
            else:
                if ext == "auto":
                    extension = extension[1:]
                else:
                    extension = ext
                if img_mode == "RGBA":
                    extension = "png"
                if suffix == "":
                    save_path = os.path.join(output_img, f"{imgname}.{extension}")
                else:
                    save_path = os.path.join(output_img, f"{imgname}_{suffix}.{extension}")
                cv2.imwrite(save_path, output)
                print(f"Up resolution for image {imgname} done and output image save in {save_path}")
        del upsampler
        torch.cuda.empty_cache()
    except:
        del upsampler
        torch.cuda.empty_cache()
