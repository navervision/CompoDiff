"""
CompoDiff
Copyright (c) 2023-present NAVER Corp.
Apache-2.0
"""
import torch
from transformers import PreTrainedModel, PretrainedConfig, CLIPTokenizer, CLIPImageProcessor
try:
    from .models import build_compodiff, build_clip
except:
    from models import build_compodiff, build_clip


class CompoDiffConfig(PretrainedConfig):
    model_type = "CompoDiff"

    def __init__(
            self,
            embed_dim: int = 768,
            model_depth: int = 12,
            model_dim: int = 64,
            model_heads: int = 16,
            timesteps: int = 1000,
            **kwargs,
            ):
        self.embed_dim = embed_dim
        self.model_depth = model_depth
        self.model_dim = model_dim
        self.model_heads = model_heads
        self.timesteps = timesteps
        super().__init__(**kwargs)


class CompoDiffModel(PreTrainedModel):
    config_class = CompoDiffConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = build_compodiff(
                config.embed_dim,
                config.model_depth,
                config.model_dim,
                config.model_heads,
                config.timesteps,
                )

    def _init_weights(self, module):
        pass

    def sample(self, image_cond, text_cond, negative_text_cond, input_mask, num_samples_per_batch=4, cond_scale=1., timesteps=None, random_seed=None):
        return self.model.sample(image_cond, text_cond, negative_text_cond, input_mask, num_samples_per_batch, cond_scale, timesteps, random_seed)


def build_model(model_name='navervision/CompoDiff-Aesthetic'):
    tokenizer = CLIPTokenizer.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

    size_cond = {'shortest_edge': 224}
    preprocess = CLIPImageProcessor(crop_size={'height': 224, 'width': 224},
                                    do_center_crop=True,
                                    do_convert_rgb=True,
                                    do_normalize=True,
                                    do_rescale=True,
                                    do_resize=True,
                                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                                    image_std=[0.26862954, 0.26130258, 0.27577711],
                                    resample=3,
                                    size=size_cond,
                                    )
    compodiff = CompoDiffModel.from_pretrained(model_name)

    clip_model = build_clip()

    return compodiff, clip_model, preprocess, tokenizer


if __name__ == '__main__':
    #''' # convert CompoDiff
    compodiff_config = CompoDiffConfig()

    compodiff = CompoDiffModel(compodiff_config)
    compodiff.model.load_state_dict(torch.load('/data/data_zoo/logs/stage2_arch.depth12-heads16_lr1e-4_text-bigG_add-art-datasets/checkpoints/model_000710000.pt')['ema_model'])
    compodiff_config.save_pretrained('/data/CompoDiff_HF')
    compodiff.save_pretrained('/data/CompoDiff_HF')
    #'''
    #compodiff, clip_model, preprocess_img, tokenizer = build_model()
