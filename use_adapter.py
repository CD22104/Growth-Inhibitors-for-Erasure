import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from attn_utils import AttentionStore, AttentionInhibit
import generate_utils as g_utils
from tqdm.notebook import tqdm

class Adapter(nn.Module):
    def __init__(self,input_number):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_number, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def get_scale(input_number,cross_information,model_path,device):
    if input_number==5:
        cross_information = cross_information[5:10]
    X_test = torch.tensor(cross_information, dtype=torch.float32)
    X_test = X_test.to(device)

    model = Adapter(input_number).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        scale = model(X_test)

    return scale

@torch.no_grad()
def use_adapter(
    input_number,
    model,
    adapter_path,
    scale_controller,
    nude_context,
    context,
    latents,
    prompts, 
    erase_targets,
    tokenizer,
    device,
    guidance_scale: float = 7.5,
):
    scale_erase_controller = AttentionStore()
    g_utils.register_attention_control(model, scale_controller,scale_erase_controller,nude_context)
    step_count = 0
    for t in model.scheduler.timesteps:
        step_count += 1
        if step_count <= 1:
            latents = g_utils.diffusion_step(model, scale_controller, latents, context, t, guidance_scale)
        else:
            break

    result = [[scale_controller.data[0][j][:,:,:,i].mean().item() for j in range(16)] for i in range(scale_controller.data[0][0].size(-1))]
    erase_scale = [get_scale(input_number, res, adapter_path,device).item() for res in result]
    print("erase_scale", erase_scale)
    equalizer = g_utils.get_equalizer(prompts, erase_targets, erase_scale, tokenizer, device)
    return equalizer