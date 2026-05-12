"""
CPL-MoE: CPL with Query-Guided Mixture of Experts

This module extends CPL (Contrastive Proposal Learning) with a Query-Guided MoE
for weakly-supervised video temporal grounding. Instead of using a single FFN
to predict proposals for all types of queries, we use specialized experts that
are selected based on the query semantics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
from models.modules.query_guided_moe import QueryGuidedMoE, QueryGuidedMoESimple
import math


class CPL_MoE(nn.Module):
    """
    CPL with Query-Guided Mixture of Experts.
    
    Key differences from original CPL:
    1. Uses Query-Guided MoE instead of single fc_gauss layer
    2. Routes to different experts based on query semantics
    3. Includes auxiliary load balancing loss for expert utilization
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']
        self.hidden_size = config['hidden_size']

        # Feature projection layers
        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        
        # Learnable special tokens
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        # Dual Transformer for cross-modal interaction
        self.trans = DualTransformer(**config['DualTransformer'])
        
        # Semantic completion head
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        
        # MoE configuration
        moe_config = config.get('MoE', {})
        self.use_moe = moe_config.get('use_moe', True)
        self.use_simple_moe = moe_config.get('use_simple_moe', False)
        
        if self.use_moe:
            if self.use_simple_moe:
                # Simplified MoE for efficiency
                self.proposal_moe = QueryGuidedMoESimple(
                    hidden_size=config['hidden_size'],
                    num_props=self.num_props,
                    num_experts=moe_config.get('num_experts', 4),
                    top_k=moe_config.get('top_k', 2),
                    dropout=config['dropout'],
                )
            else:
                # Full Query-Guided MoE
                self.proposal_moe = QueryGuidedMoE(
                    hidden_size=config['hidden_size'],
                    num_props=self.num_props,
                    num_experts=moe_config.get('num_experts', 8),
                    num_shared_experts=moe_config.get('num_shared_experts', 2),
                    top_k=moe_config.get('top_k', 2),
                    dropout=config['dropout'],
                    use_load_balance_loss=moe_config.get('use_load_balance_loss', True),
                    load_balance_weight=moe_config.get('load_balance_weight', 0.01),
                    use_2layer_gate=moe_config.get('use_2layer_gate', True),
                )
        else:
            # Fallback to original FFN
            self.fc_gauss = nn.Linear(config['hidden_size'], self.num_props * 2)
        
        # Query representation aggregator
        self.query_aggregator = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size'])
        )
        
        # Positional embedding for words
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        
        # Add prediction token to frame features
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        # Process word features
        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # Cross-modal encoding: video attends to query
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        
        # Extract multimodal representation (last token output)
        multimodal_feat = h[:, -1]  # [bsz, hidden_size]
        
        # Aggregate query representation
        # Use attention-weighted pooling over query tokens
        query_feat = self._aggregate_query(words_feat, words_mask)  # [bsz, hidden_size]
        
        # Generate Gaussian parameters using MoE or FFN
        if self.use_moe:
            gauss_param, aux_loss = self.proposal_moe(
                multimodal_feat, 
                query_feat,
                return_aux_loss=self.training
            )
            gauss_center = gauss_param[:, 0]
            gauss_width = gauss_param[:, 1]
        else:
            gauss_param = torch.sigmoid(self.fc_gauss(multimodal_feat)).view(bsz * self.num_props, 2)
            gauss_center = gauss_param[:, 0]
            gauss_width = gauss_param[:, 1]
            aux_loss = None

        # Downsample for efficiency
        props_len = n_frames // 4
        keep_idx = torch.linspace(0, n_frames - 1, steps=props_len).long()
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz * self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)
        
        # Semantic completion with masked words
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz * self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz * self.num_props, words_mask1.size(1), -1)

        pos_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, 
                                       decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)

        # Negative proposal mining
        if self.use_negative:
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(
                props_len, gauss_center, gauss_width, kwargs['epoch']
            )
            
            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1, 
                                    decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
  
            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1, 
                                    decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)

            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'aux_loss': aux_loss,  # MoE load balancing loss
        }
    
    def _aggregate_query(self, words_feat, words_mask):
        """
        Aggregate word features into a single query representation.
        Uses attention-weighted pooling based on word importance.
        """
        # Simple mean pooling with masking
        if words_mask is not None:
            mask_expanded = words_mask.unsqueeze(-1).float()
            pooled = (words_feat * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            pooled = words_feat.mean(dim=1)
        
        # Apply aggregator network
        return self.query_aggregator(pooled)
    
    def generate_gauss_weight(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))

        return weight / weight.max(dim=-1, keepdim=True)[0]

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma / 2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w / w1 * torch.exp(-(pos - c) ** 2 / (2 * w1 ** 2))
            return y1 / y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center - width / 2, min=0)
        left_center = left_width * min(epoch / self.max_epoch, 1) ** self.gamma * 0.5
        right_width = torch.clamp(1 - center - width / 2, min=0)
        right_center = 1 - right_width * min(epoch / self.max_epoch, 1) ** self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1 - right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        return int(1e5)
