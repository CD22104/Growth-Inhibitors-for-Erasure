from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import generate_utils as g_utils
import seq_aligner
from attn_utils import AttentionStore, AttentionInhibit, AttentionErase
from PIL import Image
import os
import csv
import pandas as pd
import argparse

def main(NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, MAX_NUM_WORDS, input_number, model_path, adapter_path, device, 
         prompts_path, save_folder):

    ldm_stable = StableDiffusionPipeline.from_pretrained(model_path).to(device)
    tokenizer = ldm_stable.tokenizer

    os.makedirs(save_folder, exist_ok=True)

    df = pd.read_csv(prompts_path)
    start_index = 0
    for index, row in df.iloc[start_index:].iterrows():
        print(index)
        base_prompts = str(row.prompt)
        word = str(row.word)
        case_number = str(row.case_number)
        base_prompts = [base_prompts]*2
        erase_targets = [str(row.erase_target)]
        g_cpu = torch.Generator().manual_seed(row.evaluation_seed)
        prompts = g_utils.get_inject(base_prompts,erase_targets)
        equalizer = equalizer = torch.ones(1, 77).to(device)

        cd_controller = AttentionStore()
        controller = AttentionInhibit(prompts, erase_targets, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                self_replace_steps=.4, equalizer=equalizer, tokenizer=tokenizer, device=device)
        scale_controller = AttentionErase(prompts, erase_targets, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                self_replace_steps=.4, equalizer=equalizer, tokenizer=tokenizer, device=device)
    
        images, x_t = g_utils.text2image_ldm_stable(ldm_stable, adapter_path, tokenizer,device, prompts, 
                                erase_targets, controller,cd_controller, scale_controller, input_number,
                                latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, 
                                guidance_scale=GUIDANCE_SCALE, generator=g_cpu)

        pil_image = Image.fromarray(images[1].astype('uint8'))
        pil_image.save(save_folder+'/'+str(case_number)+'_'+str(row.erase_target)+'.png')

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Img Generation')
    parser.add_argument('--NUM_DIFFUSION_STEPS', type=int, default=40, help='inference steps')
    parser.add_argument('--GUIDANCE_SCALE', type=float, default=7.5, help='CFG guidance scale')
    parser.add_argument('--MAX_NUM_WORDS', type=int, default=77, help='MAX_NUM_WORDS')
    parser.add_argument('--input_number', type=int, default=16, help='input number')
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4', help='diffusers model id or local model path')
    parser.add_argument('--adapter_path', type=str, default='model/mlp_model_final.pth', help='adapter model path')
    parser.add_argument('--device', type=str, default='cuda:1', help='cuda num')
    parser.add_argument('--prompts_path', type=str, default='I2P.csv', help='csv path')
    parser.add_argument('--save_folder',type=str, default='res')

    args = parser.parse_args()
    main(args.NUM_DIFFUSION_STEPS, args.GUIDANCE_SCALE, args.MAX_NUM_WORDS, args.input_number, args.model_path, args.adapter_path, args.device, 
          args.prompts_path, args.save_folder)

