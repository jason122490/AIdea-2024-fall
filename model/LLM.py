import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from layers.Embed import DataEmbedding_inverted, DataEmbedding, TokenEmbedding
import math

class MarkEmbedding(nn.Module):
    def __init__(self, d_model):
        super(MarkEmbedding, self).__init__()

        month_size = 13
        day_size = 32
        weekday_size = 7
        hour_size = 24
        minute_size = 60
        location_size = 18

        self.month_embed = nn.Embedding(month_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.minute_embed = nn.Embedding(minute_size, d_model)
        self.location_embed = nn.Embedding(location_size, d_model)

    def forward(self, x):
        x = x.long()
        
        month_x = self.month_embed(x[:, :, 0])
        day_x = self.day_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        hour_x = self.hour_embed(x[:, :, 3])
        minute_x = self.minute_embed(x[:, :, 4])
        location_x = self.location_embed(x[:, :, 5])

        return month_x + day_x + weekday_x + hour_x + minute_x + location_x

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_vars = args.num_vars       
        self.use_norm = args.use_norm

        self.patch_len = args.patch_len
        self.stride = args.stride
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.seq_patch_num = 1 if self.seq_len < self.patch_len else math.ceil((self.seq_len - self.patch_len) / self.stride + 1)
        self.seq_pad = self.patch_len + self.stride * (self.seq_patch_num - 1) - self.seq_len
        self.patch_num = self.seq_patch_num + self.pred_len
        
        if args.pretrained:
            model = AutoModel.from_pretrained(args.model_name)
            print('Loading pre-trained model from', args.model_name)
        else:
            config = AutoConfig.from_pretrained(args.model_name)
            model = AutoModel.from_config(config)
            print('Loading model from', args.model_name)
        
        self.embed_dim = model.config.hidden_size
        
        self.data_embedding = nn.Linear(self.num_vars, self.embed_dim)
        self.mark_embeddeing = MarkEmbedding(self.embed_dim)
        self.patch_embedding = nn.Linear(self.embed_dim * self.patch_len, self.embed_dim)

        if args.freeze:
            for param in model.parameters():
                param.requires_grad = False
        
        self.rotary_emb = model.rotary_emb
        self.layers = model.layers

        self.norm = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(0.1)
        self.proj = nn.Linear(self.embed_dim, 1)

    def forward(self, x, x_mark, pred, pred_mark):
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x.mean(1, keepdim=True).detach()
        #     x = x - means
        #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x /= stdev

        # x: [Batch, seq_len, num_var]
        x = self.data_embedding(x) + self.mark_embeddeing(x_mark)
        x = F.pad(input=x, pad=(0, 0, 0, self.seq_pad), mode='constant', value=0)
        x = x.unfold(dimension = 1, size = self.patch_len, step = self.stride)
        x = x.reshape(-1, self.seq_patch_num, self.embed_dim * self.patch_len)
        x = self.patch_embedding(x)

        # pred: [Batch, pred_len, num_var]
        pred = self.data_embedding(pred) + self.mark_embeddeing(pred_mark)

        x = torch.cat([x, pred], 1)
        B, N, C = x.shape

        position_ids = torch.arange(N, device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        empty_mask = torch.zeros(N, N).unsqueeze(0).unsqueeze(0).to(x.device)

        for layer in self.layers:
            x = layer(
                x,
                attention_mask=empty_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )[0]
        
        x = self.norm(x)
        x = self.drop(x)
        out = self.proj(x)
        out = out[:, -self.pred_len:]

        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     out = out * stdev[:, -1, :].unsqueeze(1)
        #     out = out + means[:, -1, :].unsqueeze(1)

        return out