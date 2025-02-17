# Growth-Inhibitors-for-Erasure
Our paper “Growth Inhibitors for Suppressing Inappropriate Image Concepts in Diffusion Models” has been accepted by ICLR2025!


## Methods
we propose a novel approach based on _G_rowth _I_nhibitors for _E_rasure (GIE), which can suppress inappropriate features in the image space without fine-tuning. During the diffusion process, we identify and extract features relevant to the target concept to be erased, re-weighting them to synthesize growth inhibitors. Then, we inject these features into the attention map group of the prompt so that they can be precisely transformed into appropriate ones.
![attention_00](https://github.com/user-attachments/assets/16a46d50-62cf-4bef-88e6-b7a478bb15c5)

## Installation and Usage Guide
```
git clone https://github.com/CD22104/Growth-Inhibitors-for-Erasure.git
cd Growth-Inhibitors-for-Erasure
pip install -r requirements.txt
python generate.py --model_path 'CompVis/stable-diffusion-v1-4' --adapter_path 'model/mlp_model_final.pth' --device 'cuda:0' --prompts_path 'prompts.csv' --save_folder 'output'
```

### Results for NSFW Erasure
![implicit_00](https://github.com/user-attachments/assets/583b9999-7f9a-403a-8bad-42dcd4669e72)


