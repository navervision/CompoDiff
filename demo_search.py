"""
CompoDiff
Copyright (c) 2023-present NAVER Corp.
Apache-2.0
"""
import os
import numpy as np
import base64
import requests
import json
import time
import torch
import torch.nn.functional as F
import gradio as gr
from clip_retrieval.clip_client import ClipClient
import types
from typing import Union, List, Optional, Callable
import torch
from diffusers import UnCLIPImageVariationPipeline
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from PIL import Image
import compodiff


def load_models():
    ### build model
    print("\tbuilding CompoDiff")

    compodiff_model, clip_model, img_preprocess, tokenizer = compodiff.build_model()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    compodiff_model, clip_model = compodiff_model.to(device), clip_model.to(device)

    if device != 'cpu':
        clip_model = clip_model.half()

    model_dict = {}
    model_dict['compodiff'] = compodiff_model
    model_dict['clip_model'] = clip_model
    model_dict['img_preprocess'] = img_preprocess
    model_dict['tokenizer'] = tokenizer
    model_dict['device'] = device

    return model_dict


@torch.no_grad()
def l2norm(features):
    return features / features.norm(p=2, dim=-1, keepdim=True)


def predict(images, input_text, negative_text, step, cfg_image_scale, cfg_text_scale, do_generate, source_mixing_weight):
    '''
    image_source, text_input, negative_text_input, mask_text_input, steps_input, cfg_scale, do_generate, cfg_attn_target
    '''
    device = model_dict['device']

    step = int(step)
    step = step + 1 if step < 1000 else step

    cfg_scale = (cfg_image_scale, cfg_text_scale)

    text = input_text

    if images is None:
        # t2i
        cfg_scale = (1.0, cfg_text_scale)
        text_token_dict = model_dict['tokenizer'](text=text, return_tensors='pt', padding='max_length', truncation=True)
        text_tokens, text_attention_mask = text_token_dict['input_ids'].to(device), text_token_dict['attention_mask'].to(device)

        negative_text_token_dict = model_dict['tokenizer'](text=negative_text, return_tensors='pt', padding='max_length', truncation=True)
        negative_text_tokens, negative_text_attention_mask = negative_text_token_dict['input_ids'].to(device), text_token_dict['attention_mask'].to(device)

        with torch.no_grad():
            image_cond = torch.zeros([1,1,768]).to(device)
            text_cond = model_dict['clip_model'].encode_texts(text_tokens, text_attention_mask)
            negative_text_cond = model_dict['clip_model'].encode_texts(negative_text_tokens, negative_text_attention_mask)

            sampling_start = time.time()
            mask = torch.tensor(np.zeros([64, 64], dtype='float32')).to(device).unsqueeze(0)
            sampled_image_features = model_dict['compodiff'].sample(image_cond, text_cond, negative_text_cond, mask, timesteps=step, cond_scale=cfg_scale, num_samples_per_batch=2)
            sampling_end = time.time()

            sampled_image_features_org = sampled_image_features
            sampled_image_features = l2norm(sampled_image_features)
    else:
        # CIR
        image_source = images['image'].resize((512, 512))
        mask = images['mask'].resize((512, 512))
        mask = model_dict['img_preprocess'](mask, do_normalize=False, return_tensors='pt')['pixel_values']
        mask = mask[:,:1,:,:]

        ## preprocess
        image_source = model_dict['img_preprocess'](image_source, return_tensors='pt')['pixel_values'].to(device)

        mask = (mask > 0.5).float().to(device)
        image_source = image_source * (1 - mask) 

        text_token_dict = model_dict['tokenizer'](text=text, return_tensors='pt', padding='max_length', truncation=True)
        text_tokens, text_attention_mask = text_token_dict['input_ids'].to(device), text_token_dict['attention_mask'].to(device)

        negative_text_token_dict = model_dict['tokenizer'](text=negative_text, return_tensors='pt', padding='max_length', truncation=True)
        negative_text_tokens, negative_text_attention_mask = negative_text_token_dict['input_ids'].to(device), text_token_dict['attention_mask'].to(device)

        with torch.no_grad():
            image_cond = model_dict['clip_model'].encode_images(image_source)

            text_cond = model_dict['clip_model'].encode_texts(text_tokens, text_attention_mask)

            negative_text_cond = model_dict['clip_model'].encode_texts(negative_text_tokens, negative_text_attention_mask)

            sampling_start = time.time()
            mask = transforms.Resize([64, 64])(mask)[:,0,:,:]
            mask = (mask > 0.5).float()
            if torch.sum(mask).item() == 0.0:
                mask = torch.tensor(np.zeros([64, 64], dtype='float32')).to(device).unsqueeze(0)
            sampled_image_features = model_dict['compodiff'].sample(image_cond, text_cond, negative_text_cond, mask, timesteps=step, cond_scale=cfg_scale, num_samples_per_batch=2)
            sampling_end = time.time()

            sampled_image_features_org = (1 - source_mixing_weight) * sampled_image_features + source_mixing_weight * image_cond[0]
            sampled_image_features = l2norm(sampled_image_features_org)

    if do_generate and image_decoder is not None:
        images = image_decoder(image_embeddings=sampled_image_features_org.half(), num_images_per_prompt=2).images
    else:
        images = [Image.fromarray(np.zeros([256,256,3], dtype='uint8'))]

    do_list = [['KNN results', sampled_image_features], 
              ]

    output = ''
    top1_list = []
    search_start = time.time()
    for name, features in do_list:
        results = client.query(embedding_input=features[0].tolist())[:15]
        output += f'<details open><summary>{name} outputs</summary>\n\n'
        for idx, result in enumerate(results):
            image_url = result['url']
            if idx == 0:
                top1_list.append(f'{image_url}')
            output += f'![image]({image_url})\n'

        output += '\n</details>\n\n'

    search_end = time.time()

    return output, images


if __name__ == "__main__":
    global model_dict, client, image_decoder

    model_dict = load_models()

    if 'cuda' in model_dict['device']:
        image_decoder = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16).to('cuda:0')
    else:
        image_decoder = None

    client = ClipClient(url="https://knn.laion.ai/knn-service",
                        indice_name="laion5B-L-14",
                        )

    ### define gradio demo
    title = 'CompoDiff demo'

    md_title = f'''# {title}
    Diffusion on {model_dict["device"]}, K-NN Retrieval using https://rom1504.github.io/clip-retrieval.
    '''
    md_below = f'''### Tips:
    Here are some tips for using the demo:
    + If you want to apply more of the original image's context, increase the source weight in the Advanced options from 0.1. This will convey the context of the original image as a strong signal.
    + If you want to exclude specific keywords, you can add them to the Negative text input.
    + Try using "generate image with unCLIP" to create images. You can see some interesting generated images that are as fascinating as search results.
    + If you only input text and no image, it will work like the prior of unCLIP.
    '''


    with gr.Blocks(title=title) as demo:
        gr.Markdown(md_title)
        with gr.Row():
            with gr.Column():
                image_source = gr.Image(type='pil', label='Source image', tool='sketch')
                with gr.Row():
                    steps_input = gr.Radio(['2', '3', '5', '10'], value='10', label='denoising steps')
                    if model_dict['device'] == 'cpu':
                        do_generate = gr.Checkbox(value=False, label='generate image with unCLIP', visible=False)
                    else:
                        do_generate = gr.Checkbox(value=False, label='generate image with unCLIP', visible=True)
                with gr.Accordion('Advanced options', open=False):
                    with gr.Row():
                        cfg_image_scale = gr.Number(value=1.5, label='image condition scale')
                        cfg_text_scale = gr.Number(value=7.5, label='text condition scale')
                    source_mixing_weight = gr.Number(value=0.1, label='source weight (0.0~1.0)')
                text_input = gr.Textbox(value='', label='Input text guidance')
                negative_text_input = gr.Textbox(value='', label='Negative text') # low quality, text overlay, logo
                submit_button = gr.Button('Submit')
                gr.Markdown(md_below)
            with gr.Column():
                if model_dict['device'] == 'cpu':
                    gallery = gr.Gallery(label='Generated images', visible=False).style(grid=[2])
                else:
                    gallery = gr.Gallery(label='Generated images', visible=True).style(grid=[2])
                md_output = gr.Markdown(label='Output')
        submit_button.click(predict, inputs=[image_source, text_input, negative_text_input, steps_input, cfg_image_scale, cfg_text_scale, do_generate, source_mixing_weight], outputs=[md_output, gallery])
    demo.launch(server_name='0.0.0.0',
                server_port=8000)

