import torch

from typing import Union

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
from shap_e.util.image_util import load_image

from symai import Expression


class ShapE(Expression):
    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xm          = load_model('transmitter', device=self.device)
        self.image_model = load_model('image300M', device=self.device)
        self.text_model  = load_model('text300M', device=self.device)
        self.diffusion   = diffusion_from_config(load_config('diffusion'))

    def forward(self, query: str, render_mode: Union['nerf', 'stf'] = 'nerf', *args, **kwargs):
        batch_size = 1
        guidance_scale = 3.0
        query = query.lower().strip()

        # check if the query is a path to an image or a string prompt
        if query.endswith('.jpg') or query.endswith('.png') or query.endswith('.jpeg'):
            # To get the best result, you should remove the background and show only the object of interest to the model.
            image = load_image(query)
            model_kwargs = dict(images=[image] * batch_size)
            model = self.image_model
        else:
            model_kwargs = dict(texts=[query] * batch_size)
            model = self.text_model

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        size = 64 # this is the size of the renders; higher values take longer to render.
        cameras = create_pan_cameras(size, self.device)
        images  = decode_latent_images(self.xm, latents[0], cameras, rendering_mode=render_mode)
        return images
