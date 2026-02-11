import os
import math
import numpy as np
import torch
import safetensors.torch as sf
import cv2

from PIL import Image, ImageOps
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from torch.hub import download_url_to_file
import argparse
from tqdm import tqdm


# ----------------------------
# Hardcoded prompt + hyperparams (keep same as now)
# ----------------------------
# NOTE: STOPPED this hardcode!
# TARGET_W = 512
# TARGET_H = 512

SEED  = 42
STEPS = 25
CFG   = 2.0
A_PROMPT = "best quality"
N_PROMPT = "lowres, bad anatomy, bad hands, cropped, worst quality"

# ----------------------------
# Helpers (from IC-Light style)
# ----------------------------
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)  # B,C,H,W
    return h


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)  # H,W,C
        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
        results.append(y)
    return results


def resize_without_crop(image_uint8, target_w, target_h):
    pil = Image.fromarray(image_uint8)
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil)


def bbox_crop_by_mask(image_rgb_uint8, mask_uint8):
    pil_img  = Image.fromarray(image_rgb_uint8).convert("RGB")
    pil_mask = Image.fromarray(mask_uint8).convert("L")

    # NOTE: assuming foreground is black → invert so FG becomes non-zero
    pil_mask = ImageOps.invert(pil_mask)

    bbox = pil_mask.getbbox()
    if bbox:
        pil_img  = pil_img.crop(bbox)
        pil_mask = pil_mask.crop(bbox)

    crop_sizes = pil_img.size  # (W,H) AFTER cropping
    return np.array(pil_img), np.array(pil_mask), bbox, crop_sizes


@torch.inference_mode()
def encode_prompt_pair(tokenizer, text_encoder, device, positive_prompt, negative_prompt):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    def encode_inner(txt: str):
        tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i:i+chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]
        token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
        return text_encoder(token_ids).last_hidden_state

    c  = encode_inner(positive_prompt)
    uc = encode_inner(negative_prompt)

    c_len  = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat  = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c  = torch.cat([c]  * c_repeat,  dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c  = torch.cat([p[None, ...] for p in c],  dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


# ----------------------------
# Load IC-Light (text-conditioned fg model)
# ----------------------------
sd15_name = "stablediffusionapi/realistic-vision-v51"
tokenizer    = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae          = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet         = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(
        8,
        unet.conv_in.out_channels,
        unet.conv_in.kernel_size,
        unet.conv_in.stride,
        unet.conv_in.padding,
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs["cross_attention_kwargs"] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

model_path = "./models/iclight_sd15_fc.safetensors"
os.makedirs("./models", exist_ok=True)
if not os.path.exists(model_path):
    download_url_to_file(
        url="https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
        dst=model_path,
    )

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged

device = torch.device("cuda")
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae          = vae.to(device=device, dtype=torch.bfloat16)
unet         = unet.to(device=device, dtype=torch.float16)

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1,
)

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None,
).to(device)


# ----------------------------
# Args (ONLY dirs; no extra knobs)
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relight_type",
        type=str,
        required=True,
        choices=["noon_sunlight_1", "golden_sunlight_1", "foggy_1", "moonlight_1"],
    )
    parser.add_argument("--input_dir", type=str, default="/home/shenzhen/Datasets/VITON/test/image")
    parser.add_argument("--mask_dir", type=str, default="/home/shenzhen/Datasets/VITON/test/fg_masks")
    parser.add_argument("--out_root", type=str, default="./outputs")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    return parser.parse_args()


# ----------------------------
# Main: folder inference
# ----------------------------
def main():
    args = parse_args()

    # NOTE: read target width and height from args
    TARGET_W = args.image_width
    TARGET_H = args.image_height

    relighting_prompts_6 = {
        "noon_sunlight_1": "Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood.",
        "golden_sunlight_1": "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood.",
        "foggy_1": "Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood.",
        "moonlight_1": "Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood.",
    }

    PROMPT = relighting_prompts_6[args.relight_type]
    input_dir = args.input_dir
    mask_dir = args.mask_dir
    output_dir = f"{args.out_root}/{TARGET_W}x{TARGET_H}/{args.relight_type}/VITON/test/image"
    os.makedirs(output_dir, exist_ok=True)

    fnames = sorted(os.listdir(input_dir))
    print(f"\n==== Running relight: {args.relight_type} on CUDA={os.environ.get('CUDA_VISIBLE_DEVICES','ALL')} ====\n")
    print(f"Input : {input_dir}")
    print(f"Mask  : {mask_dir}")
    print(f"Output: {output_dir}\n")

    # Text embeddings (reuse across images because prompt is constant)
    conds, unconds = encode_prompt_pair(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        positive_prompt=PROMPT + ", " + A_PROMPT,
        negative_prompt=N_PROMPT,
    )

    rng = torch.Generator(device=device).manual_seed(int(SEED))

    for fname in tqdm(fnames, desc=f"Relighting [{args.relight_type}]"):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        base = os.path.splitext(fname)[0]
        img_path = os.path.join(input_dir, fname)
        mask_path = os.path.join(mask_dir, base + ".png")
        if not os.path.exists(mask_path):
            continue

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        # 1) Crop to FG bbox
        crop_rgb, _, bbox, crop_size = bbox_crop_by_mask(img_rgb, m)
        if bbox is None:
            continue

        # 2) Resize cropped region to 512x512
        in_512 = resize_without_crop(crop_rgb, TARGET_W, TARGET_H)

        # 3) concat_conds = VAE(fg_512) latent
        fg_tensor = numpy2pytorch([in_512]).to(device=vae.device, dtype=vae.dtype)
        fg_latent = vae.encode(fg_tensor).latent_dist.mode() * vae.config.scaling_factor  # (1,4,h,w)

        # 4) Generate
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=TARGET_W,
            height=TARGET_H,
            num_inference_steps=int(STEPS),
            num_images_per_prompt=1,
            generator=rng,
            output_type="latent",
            guidance_scale=float(CFG),
            cross_attention_kwargs={"concat_conds": fg_latent},
        ).images.to(vae.dtype) / vae.config.scaling_factor

        # 5) Decode + resize back to crop size
        out_pixels = vae.decode(latents).sample
        out_512 = pytorch2numpy(out_pixels, quant=True)[0]
        out_restore = resize_without_crop(out_512, crop_size[0], crop_size[1])

        # Save final crop output
        out_bgr = cv2.cvtColor(out_restore, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, fname), out_bgr)


if __name__ == "__main__":
    main()