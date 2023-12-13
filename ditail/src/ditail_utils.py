# credits: https://github.com/MichalGeyer/pnp-diffusers/blob/main/pnp_utils.py

import os
import torch
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# def get_save_dir(output_dir, img_path):
#     os.makedirs(output_dir, exist_ok=True)
#     file = os.path.basename(img_path)
#     indices = [d for d in os.listdir(output_dir) if d.startswith(file)]
#     return os.path.join(output_dir, f'{file}_{len(indices)}')

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)

def register_attn_inj(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(q.shape[0] // 3)
                # inject pos chunk
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject neg chunk
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)
        return forward
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def register_conv_inj(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb, scale):
            hidden_states = input_tensor
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)
            if self.upsample is not None:
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor, scale=scale)
                hidden_states = self.upsample(hidden_states, scale=scale)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor, scale=scale)
                hidden_states = self.downsample(hidden_states, scale=scale)
            hidden_states = self.conv1(hidden_states, scale)
            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.nonlinearity(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states, scale)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject pos chunk
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject neg chunk
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]
            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor, scale)
            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
            return output_tensor
        return forward
    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)