import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from layers.Embed import DataEmbedding_inverted, DataEmbedding

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.use_norm = args.use_norm
        self.embed_mode = args.embed_mode

        self.patch_len = 16
        self.stride = 8
        self.patch_num = (args.seq_len - self.patch_len) // self.stride + 1
        
        if args.pretrained:
            model = AutoModel.from_pretrained(args.model_name)
            print('Loading pre-trained model from', args.model_name)
        else:
            config = AutoConfig.from_pretrained(args.model_name)
            model = AutoModel.from_config(config)
            print('Loading model from', args.model_name)

        self.embed_dim = model.config.hidden_size

        if args.embed_mode == 'Inverted':
            self.embedding = DataEmbedding_inverted(args.seq_len, self.embed_dim, args.embed, args.freq, args.dropout)
        elif args.embed_mode == 'Patch':
            self.embedding = DataEmbedding(self.patch_len, self.embed_dim)
        else:
            raise ValueError('embed_mode should be Inverted or Patch')
        
        if args.freeze:
            for param in model.parameters():
                param.requires_grad = False
        
        self.rotary_emb = model.rotary_emb
        self.layers = model.layers

        self.norm = nn.LayerNorm(self.embed_dim)

        self.drop = nn.Dropout(0.1)
        if args.embed_mode == 'Inverted':
            self.proj = nn.Linear(self.embed_dim, args.pred_len)
        elif args.embed_mode == 'Patch':
            self.proj = nn.Linear(self.patch_num * self.embed_dim, args.pred_len)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        num_var = x_enc.shape[-1]

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.embed_mode == 'Patch':
            x_enc = x_enc.permute(0, 2, 1)
            x_enc = x_enc.unfold(dimension = -1, size = self.patch_len, step = self.stride)
            x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], x_enc.shape[2], x_enc.shape[3])

        enc_out = self.embedding(x_enc, x_mark_enc)
        B, N, C = enc_out.shape

        position_ids = torch.arange(N, device=enc_out.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(enc_out, position_ids)
        empty_mask = torch.zeros(N, N).unsqueeze(0).unsqueeze(0).to(enc_out.device)

        for layer in self.layers:
            enc_out = layer(
                enc_out,
                attention_mask=empty_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )[0]
        
        enc_out = self.norm(enc_out)
        if self.embed_mode == 'Patch':
            enc_out = enc_out.view(B, N, -1)
        
        enc_out = self.drop(enc_out)
        dec_out = self.proj(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :num_var]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out