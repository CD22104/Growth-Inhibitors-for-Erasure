# Growth-Inhibitors-for-Erasure

## Methods
we propose a novel approach based on _G_rowth _I_nhibitors for _E_rasure (GIE), which can suppress inappropriate features in the image space without fine-tuning. During the diffusion process, we identify and extract features relevant to the target concept to be erased, re-weighting them to synthesize growth inhibitors. Then, we inject these features into the attention map group of the prompt so that they can be precisely transformed into appropriate ones.
![attention_00](https://github.com/user-attachments/assets/16a46d50-62cf-4bef-88e6-b7a478bb15c5)

## Result

### Results for NSFW Erasure
![implicit_00](https://github.com/user-attachments/assets/583b9999-7f9a-403a-8bad-42dcd4669e72)

### Compare with Other Baselines
![comparation_00](https://github.com/user-attachments/assets/5e08ddcb-ce0a-4908-8322-8e270fe85b75)

### Results for Style Erasure
![style_name_00](https://github.com/user-attachments/assets/9f924552-6797-4a40-98f3-50ccd840bed4)

