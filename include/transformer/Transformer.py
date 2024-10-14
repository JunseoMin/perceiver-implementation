import torch
import torch.nn as nn

from DecoderModel import Decoder
from EncoderModel import Encoder

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Transformer(nn.Module):
    def __init__(self, n_src, n_trg, src_pad_idx, trg_pad_idx, d_invec ,d_model, d_k, d_v, d_inner, n_layer, n_head, dropout = 0.1, n_pos = 200,
                 trg_emb_weight_sharing = True, src_emb_weight_sharing = True   ):
        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.d_model = d_model

        scale_emb = True
        self.scale_proj = True

        self.d_model = d_model

        self.encoder = Encoder(n_src,d_invec,n_layer,n_head,d_k, d_v, d_model,d_inner, src_pad_idx, dropout=dropout, n_pos=n_pos, scale_embedding=scale_emb)
        self.decoder = Decoder(n_trg, d_invec, n_layer, n_head, d_k, d_v, d_model, d_inner, trg_pad_idx, n_pos,dropout, scale_emb)

        self.trg_prj = nn.Linear(d_model, n_trg, bias = False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if trg_emb_weight_sharing:
            self.trg_prj.weight = self.decoder.trg_emb.weight
        
        if src_emb_weight_sharing:
            self.encoder.src_embed.weight = self.decoder.trg_emb.weight


    def foward(self,src_seq, trg_seq):
        src_mask = get_pad_mask(src_mask, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_mask, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output = self.encoder(src_seq,src_mask)
        dec_output = self.decoder(trg_seq,trg_mask,enc_output,src_mask)

        seq_logit = self.trg_prj(dec_output)

        if self.scale_proj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit
        
