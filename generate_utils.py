import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
from use_adapter import use_adapter

def get_inject(base_prompts, erase_targets):
    join_result = " ".join(erase_targets)
    prompts = base_prompts.copy()
    if base_prompts[1][-1]==" ":
        prompts[1] = base_prompts[1] + join_result
    else:
        prompts[1] = base_prompts[1] + " " + join_result
    return prompts

def get_equalizer(prompts, erase_targets, erase_scale, tokenizer, device):
    equalizer = torch.ones(1, 77).to(device)
    pos = 0
    erase_scale = torch.tensor(erase_scale).to(device)
    for concept in erase_targets:
        split = concept.split(" ")     
        for word in split:
            inds = get_word_inds(prompts[1], word, tokenizer)
            count = len(tokenizer.encode(word)) - 2
            value = erase_scale[pos:pos + count]
            for i in range(0, len(inds), count):
                equalizer[:, inds[i]:inds[i + count-1]+1] = value       
            pos += count
    return equalizer

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    pil_img = Image.fromarray(image_)
    display(pil_img)


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def get_text_injects(controller,text_embeddings,nude_embeddings):
    text_weighted = text_embeddings

    base_embeddings = text_embeddings[1:,controller.mapper[0],:]
    insert_embeddings = nude_embeddings[:,controller.erase_mapper[0],:]

    alphas = controller.alphas[0][0].unsqueeze(-1) 
    alphas = alphas.expand(-1, 77, 768)
    text_weighted[1:,:,:] = base_embeddings * alphas + insert_embeddings * (1 - alphas)
    
    return text_weighted

@torch.no_grad()
def text2image_ldm_stable(
    model,
    adapter_path,
    tokenizer,
    device,
    prompt: List[str],
    erase_targets: List[str],
    controller,
    erase_controller, 
    scale_controller,
    input_number,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    height = width = 512
    batch_size = len(prompt)


    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    text_nude = model.tokenizer(
        [" ".join(erase_targets)],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    nude_embeddings = model.text_encoder(text_nude.input_ids.to(model.device))[0]

    text_embeddings = get_text_injects(controller,text_embeddings,nude_embeddings)
   
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    context = [text_embeddings, uncond_embeddings]
    context = torch.cat(context)

    nude_embeddings = nude_embeddings.expand(text_embeddings.shape[0], -1, -1)
    nude_context = [nude_embeddings,uncond_embeddings]
    nude_context = torch.cat(nude_context)
    
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)

    adapter_result = use_adapter(input_number, model, adapter_path, scale_controller, nude_context, context, 
                            latents, prompt, erase_targets, tokenizer,device, guidance_scale)
    controller.equalizer = adapter_result

    model.scheduler.set_timesteps(num_inference_steps)
    register_attention_control(model, controller, erase_controller, nude_context)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller,erase_controller,nude_embedding):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            if encoder_hidden_states is None:
                erase_encoder_hidden_states=None
            else:
                erase_encoder_hidden_states=nude_embedding
        
            is_cross = encoder_hidden_states is not None
            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            erase_hidden_states=hidden_states
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            erase_batch_size, erase_sequence_length, erase_ = (
                erase_hidden_states.shape if erase_encoder_hidden_states is None else erase_encoder_hidden_states.shape
            )

            erase_attention_mask=attention_mask 
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            erase_attention_mask = self.prepare_attention_mask(erase_attention_mask, erase_sequence_length, erase_batch_size)
            
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
                erase_hidden_states = self.group_norm(erase_hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            erase_query = self.to_q(erase_hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            if erase_encoder_hidden_states is None:
                erase_encoder_hidden_states =erase_hidden_states
            elif self.norm_cross:
                erase_encoder_hidden_states = self.norm_encoder_hidden_states(erase_encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            erase_key = self.to_k(erase_encoder_hidden_states)
            erase_value = self.to_v(erase_encoder_hidden_states)
            
            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            erase_query = self.head_to_batch_dim(erase_query)
            erase_key = self.head_to_batch_dim(erase_key)
            erase_value = self.head_to_batch_dim(erase_value)

            erase_attention_probs = self.get_attention_scores(erase_query, erase_key, erase_attention_mask)
            erase_attention_probs = erase_controller(erase_attention_probs, is_cross, place_in_unet,erase_attention_probs)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet,erase_attention_probs)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            hidden_states = to_out(hidden_states)
            
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward
        
    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    erase_controller.num_att_layers = cross_att_count

def diffusion_step(model, controller, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_prediction_text,noise_pred_uncond= noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if ptr < len(split_text) and cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):

    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}

    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)

    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],i)

    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words