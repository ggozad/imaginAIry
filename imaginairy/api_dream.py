import logging
import os
import re

import numpy as np
import PIL
import torch
import torch.nn
from einops import rearrange
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pytorch_lightning import seed_everything

from imaginairy.enhancers.clip_masking import get_img_mask
from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.enhancers.upscale_realesrgan import upscale_image
from imaginairy.img_utils import pillow_fit_image_within, pillow_img_to_torch_image
from imaginairy.log_utils import (
    ImageLoggingContext,
    log_conditioning,
    log_img,
    log_latent,
)
from imaginairy.model_manager import get_diffusion_model
from imaginairy.safety import SafetyMode, create_safety_score
from imaginairy.samplers.base import NoiseSchedule, get_sampler, noise_an_image
from imaginairy.schema import ImaginePrompt, ImagineResult
from imaginairy.utils import (
    fix_torch_group_norm,
    fix_torch_nn_layer_norm,
    get_device,
    platform_appropriate_autocast,
    randn_seeded,
)

logger = logging.getLogger(__name__)

# leave undocumented. I'd ask that no one publicize this flag. Just want a
# slight barrier to entry. Please don't use this is any way that's gonna cause
# the media or politicians to freak out about AI...
IMAGINAIRY_SAFETY_MODE = os.getenv("IMAGINAIRY_SAFETY_MODE", SafetyMode.STRICT)
if IMAGINAIRY_SAFETY_MODE in {"disabled", "classify"}:
    IMAGINAIRY_SAFETY_MODE = SafetyMode.RELAXED
elif IMAGINAIRY_SAFETY_MODE == "filter":
    IMAGINAIRY_SAFETY_MODE = SafetyMode.STRICT


def dream_image_files(
    prompts,
    outdir,
    precision="autocast",
    output_file_extension="jpg",
    print_caption=False,
):
    generated_imgs_path = os.path.join(outdir, "generated")
    os.makedirs(generated_imgs_path, exist_ok=True)

    base_count = len(os.listdir(generated_imgs_path))
    output_file_extension = output_file_extension.lower()
    if output_file_extension not in {"jpg", "png"}:
        raise ValueError("Must output a png or jpg")

    for result in dream(
        prompts,
        precision=precision,
        add_caption=print_caption,
    ):
        prompt = result.prompt
        img_str = ""
        if prompt.init_image:
            img_str = f"_img2img-{prompt.init_image_strength}"
        basefilename = (
            f"{base_count:06}_{prompt.seed}_{prompt.sampler_type.replace('_', '')}{prompt.steps}_"
            f"PS{prompt.prompt_strength}{img_str}_{prompt_normalized(prompt.prompt_text)}"
        )

        for image_type in result.images:
            subpath = os.path.join(outdir, image_type)
            os.makedirs(subpath, exist_ok=True)
            filepath = os.path.join(
                subpath, f"{basefilename}_[{image_type}].{output_file_extension}"
            )
            result.save(filepath, image_type=image_type)
            logger.info(f"🖼  [{image_type}] saved to: {filepath}")
        base_count += 1
        del result


def dream(
    prompts,
    precision="autocast",
    img_callback=None,
    half_mode=None,
    add_caption=False,
):
    latent_channels = 4
    downsampling_factor = 8
    batch_size = 1

    prompts = [ImaginePrompt(prompts)] if isinstance(prompts, str) else prompts
    prompts = [prompts] if isinstance(prompts, ImaginePrompt) else prompts

    try:
        num_prompts = str(len(prompts))
    except TypeError:
        num_prompts = "?"

    if get_device() == "cpu":
        logger.info("Running in CPU mode. it's gonna be slooooooow.")

    with torch.no_grad(), platform_appropriate_autocast(
        precision
    ), fix_torch_nn_layer_norm(), fix_torch_group_norm():
        for i, prompt in enumerate(prompts):
            logger.info(
                f"Generating 🖼  {i + 1}/{num_prompts}: {prompt.prompt_description()}"
            )
            model = get_diffusion_model(
                weights_location=prompt.model, half_mode=half_mode
            )
            with ImageLoggingContext(
                prompt=prompt,
                model=model,
                img_callback=img_callback,
            ):
                seed_everything(prompt.seed)
                model.tile_mode(False)

                neutral_conditioning = None
                if prompt.prompt_strength != 1.0:
                    neutral_conditioning = model.get_learned_conditioning(
                        batch_size * [""]
                    )
                    log_conditioning(neutral_conditioning, "neutral conditioning")
                if prompt.conditioning is not None:
                    positive_conditioning = prompt.conditioning
                else:
                    total_weight = sum(wp.weight for wp in prompt.prompts)
                    positive_conditioning = sum(
                        model.get_learned_conditioning(wp.text)
                        * (wp.weight / total_weight)
                        for wp in prompt.prompts
                    )
                log_conditioning(positive_conditioning, "positive conditioning")

                shape = [
                    batch_size,
                    latent_channels,
                    prompt.height // downsampling_factor,
                    prompt.width // downsampling_factor,
                ]

                sampler = get_sampler(prompt.sampler_type, model)
                t_enc = init_latent = init_latent_noised = None
                print(t_enc)
                log_latent(init_latent_noised, "init_latent_noised")

                samples = sampler.sample(
                    num_steps=prompt.steps,
                    initial_latent=init_latent_noised,
                    positive_conditioning=positive_conditioning,
                    neutral_conditioning=neutral_conditioning,
                    guidance_scale=prompt.prompt_strength,
                    t_start=t_enc,
                    orig_latent=init_latent,
                    shape=shape,
                    batch_size=1,
                )

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = x_sample.to(torch.float32)
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    x_sample_8_orig = x_sample.astype(np.uint8)
                    img = Image.fromarray(x_sample_8_orig)

                    upscaled_img = None
                    rebuilt_orig_img = None

                    if add_caption:
                        caption = generate_caption(img)
                        logger.info(f"Generated caption: {caption}")

                    safety_score = create_safety_score(
                        img,
                        safety_mode=IMAGINAIRY_SAFETY_MODE,
                    )
                    if not safety_score.is_filtered:
                        if prompt.upscale:
                            logger.info("Upscaling 🖼  using real-ESRGAN...")
                            upscaled_img = upscale_image(img)

                    yield ImagineResult(
                        img=img,
                        prompt=prompt,
                        upscaled_img=upscaled_img,
                        is_nsfw=safety_score.is_nsfw,
                        safety_score=safety_score,
                        modified_original=rebuilt_orig_img,
                    )


def prompt_normalized(prompt):
    return re.sub(r"[^a-zA-Z0-9.,\[\]-]+", "_", prompt)[:130]
