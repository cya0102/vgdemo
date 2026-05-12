"""
Query-Guided Mixture of Experts (QG-MoE) for Weakly-Supervised Video Grounding

This module implements a query-guided MoE mechanism that selects appropriate experts
based on the semantic information of the query to predict proposal segments.

Inspired by MoE++ (https://github.com/SkyworkAI/MoE-plus-plus)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import copy


class QueryGuidedRouter(nn.Module):
    """
    Query-guided router that uses query semantics to route to different experts.
    Different from standard MoE routers that use input features directly,
    this router leverages query embeddings to guide expert selection.
    """
    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        use_2layer_gate: bool = True,
        use_logits_norm: bool = False,
        gate_norm_std: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_logits_norm = use_logits_norm
        self.gate_norm_std = gate_norm_std
        
        # Query encoder to extract semantic information
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-modal fusion for routing
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Router network
        if use_2layer_gate:
            self.wg = nn.Sequential(
                nn.Linear(hidden_dim, num_experts * 4, bias=False),
                nn.Tanh(),
                nn.Linear(num_experts * 4, num_experts, bias=False)
            )
        else:
            self.wg = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Residual gate mapping for hierarchical routing
        self.gate_map = nn.Linear(num_experts, num_experts, bias=False)
        
    def forward(
        self, 
        multimodal_feat: torch.Tensor,
        query_feat: torch.Tensor,
        gate_residual: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[int, List], torch.Tensor, torch.Tensor]:
        """
        Args:
            multimodal_feat: Multi-modal features [batch_size, hidden_dim]
            query_feat: Query features [batch_size, hidden_dim]
            gate_residual: Optional residual from previous layer
        
        Returns:
            expert_info: Dict mapping expert_id to (token_indices, gates)
            gate_logits: Raw logits for load balancing loss
            expert_weights: Soft weights for all experts [batch_size, num_experts]
        """
        batch_size = multimodal_feat.size(0)
        
        # Encode query semantics
        query_encoded = self.query_encoder(query_feat)
        
        # Fuse multimodal features with query
        fused_feat = torch.cat([multimodal_feat, query_encoded], dim=-1)
        fused_feat = self.fusion_gate(fused_feat)
        
        # Compute routing logits
        logits = self.wg(fused_feat.float())
        
        # Add residual if provided (for hierarchical routing)
        if gate_residual is not None:
            gate_residual = self.gate_map(gate_residual.to(self.gate_map.weight.dtype))
            logits = logits + gate_residual
        
        # Normalize logits if needed
        if self.use_logits_norm:
            logits_std = logits.std(dim=1, keepdim=True)
            logits = logits / (logits_std / self.gate_norm_std + 1e-6)
        
        # Compute soft weights for all experts
        expert_weights = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(expert_weights, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Build expert info dictionary
        expert_info = defaultdict(list)
        for expert_id in range(self.num_experts):
            token_ids, score_ids = torch.nonzero(top_k_indices == expert_id, as_tuple=True)
            expert_info[expert_id] = [token_ids, top_k_weights[token_ids, score_ids]]
        
        return expert_info, logits, expert_weights


class ProposalExpert(nn.Module):
    """
    Expert network for proposal prediction.
    Each expert specializes in predicting proposals for different types of queries.
    """
    def __init__(self, hidden_size: int, num_props: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_props = num_props
        
        self.expert_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_props * 2)  # center and width for each proposal
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, hidden_dim]
        Returns:
            Proposal parameters [batch_size, num_props * 2]
        """
        return self.expert_net(x)


class CopyExpert(nn.Module):
    """Copy expert that returns the input unchanged."""
    def __init__(self, hidden_size: int, num_props: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_props * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ZeroExpert(nn.Module):
    """Zero expert that returns zeros."""
    def __init__(self, hidden_size: int, num_props: int):
        super().__init__()
        self.num_props = num_props
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return torch.zeros(batch_size, self.num_props * 2, device=x.device, dtype=x.dtype)


class ConstantExpert(nn.Module):
    """
    Constant expert with learnable constant and adaptive mixing.
    Useful for handling edge cases.
    """
    def __init__(self, hidden_size: int, num_props: int):
        super().__init__()
        self.num_props = num_props
        self.constant = nn.Parameter(torch.empty(num_props * 2))
        nn.init.uniform_(self.constant, 0.2, 0.8)  # Initialize in reasonable range
        
        self.wg = nn.Linear(hidden_size, 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(hidden_size, num_props * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.softmax(self.wg(x))
        transformed = self.fc(x)
        return weight[:, 0:1] * transformed + weight[:, 1:2] * self.constant.unsqueeze(0)


class QueryGuidedMoE(nn.Module):
    """
    Query-Guided Mixture of Experts for proposal prediction.
    
    This module replaces the simple FFN in CPL with a MoE architecture that
    selects experts based on query semantics. Different query types
    (e.g., action-focused, object-focused, temporal-focused) are routed
    to specialized experts.
    """
    def __init__(
        self,
        hidden_size: int,
        num_props: int,
        num_experts: int = 8,
        num_shared_experts: int = 2,  # Number of shared experts (always active)
        top_k: int = 2,
        dropout: float = 0.1,
        use_load_balance_loss: bool = True,
        load_balance_weight: float = 0.01,
        use_2layer_gate: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_props = num_props
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.use_load_balance_loss = use_load_balance_loss
        self.load_balance_weight = load_balance_weight
        
        # Query-guided router
        self.router = QueryGuidedRouter(
            query_dim=hidden_size,
            hidden_dim=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            use_2layer_gate=use_2layer_gate,
        )
        
        # Build experts: regular experts + special experts
        num_regular_experts = num_experts - 3  # Reserve 3 slots for special experts
        self.experts = nn.ModuleList()
        
        # Regular proposal experts
        for _ in range(max(1, num_regular_experts)):
            self.experts.append(ProposalExpert(hidden_size, num_props, dropout))
        
        # Special experts (similar to MoE++)
        self.experts.append(ConstantExpert(hidden_size, num_props))  # Constant expert
        self.experts.append(CopyExpert(hidden_size, num_props))       # Copy expert
        self.experts.append(ZeroExpert(hidden_size, num_props))       # Zero expert
        
        # Shared experts (always active, not routed)
        self.shared_experts = nn.ModuleList([
            ProposalExpert(hidden_size, num_props, dropout)
            for _ in range(num_shared_experts)
        ])
        
        # Output projection to combine routed and shared expert outputs
        total_expert_outputs = 1 + num_shared_experts  # 1 for routed, n for shared
        self.output_proj = nn.Linear(num_props * 2 * total_expert_outputs, num_props * 2)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self, 
        multimodal_feat: torch.Tensor,
        query_feat: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            multimodal_feat: Fused multi-modal features [batch_size, hidden_dim]
            query_feat: Query representation [batch_size, hidden_dim]
            return_aux_loss: Whether to return auxiliary load balancing loss
            
        Returns:
            gauss_params: Proposal parameters [batch_size * num_props, 2]
            aux_loss: Auxiliary load balancing loss (optional)
        """
        batch_size = multimodal_feat.size(0)
        
        # Normalize inputs
        multimodal_feat = self.layer_norm(multimodal_feat)
        
        # Get routing information
        expert_info, gate_logits, expert_weights = self.router(
            multimodal_feat, query_feat
        )
        
        # Compute routed expert outputs
        routed_output = torch.zeros(batch_size, self.num_props * 2, 
                                    device=multimodal_feat.device, 
                                    dtype=multimodal_feat.dtype)
        
        for expert_id, (token_indices, gates) in expert_info.items():
            if len(token_indices) == 0:
                continue
            
            expert = self.experts[expert_id]
            expert_input = multimodal_feat[token_indices]
            expert_output = expert(expert_input)
            
            # Weight by gate values
            weighted_output = expert_output * gates.unsqueeze(-1)
            routed_output.index_add_(0, token_indices, weighted_output)
        
        # Compute shared expert outputs (always active)
        shared_outputs = []
        for shared_expert in self.shared_experts:
            shared_output = shared_expert(multimodal_feat)
            shared_outputs.append(shared_output)
        
        # Combine routed and shared outputs
        all_outputs = [routed_output] + shared_outputs
        combined = torch.cat(all_outputs, dim=-1)
        
        # Project to final output
        output = self.output_proj(combined)
        
        # Apply sigmoid to get valid proposal parameters (center, width in [0, 1])
        gauss_params = torch.sigmoid(output).view(batch_size * self.num_props, 2)
        
        # Compute auxiliary load balancing loss
        aux_loss = None
        if return_aux_loss and self.use_load_balance_loss and self.training:
            aux_loss = self._compute_load_balance_loss(expert_weights, expert_info)
        
        return gauss_params, aux_loss
    
    def _compute_load_balance_loss(
        self, 
        expert_weights: torch.Tensor,
        expert_info: Dict[int, List]
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage balanced expert utilization.
        
        Based on Switch Transformer's load balancing loss.
        """
        batch_size = expert_weights.size(0)
        
        # Compute fraction of tokens routed to each expert
        expert_counts = torch.zeros(self.num_experts, device=expert_weights.device)
        for expert_id, (token_indices, _) in expert_info.items():
            expert_counts[expert_id] = len(token_indices)
        
        tokens_per_expert = expert_counts / (batch_size + 1e-6)
        
        # Compute mean routing probability to each expert
        router_prob_per_expert = expert_weights.mean(dim=0)
        
        # Load balance loss
        load_balance_loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()
        
        return self.load_balance_weight * load_balance_loss


class QueryGuidedMoESimple(nn.Module):
    """
    Simplified Query-Guided MoE without shared experts.
    More efficient and suitable for smaller models.
    """
    def __init__(
        self,
        hidden_size: int,
        num_props: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_props = num_props
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Simple query-conditioned router
        self.router = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_props * 2)
            )
            for _ in range(num_experts)
        ])
        
    def forward(
        self, 
        multimodal_feat: torch.Tensor,
        query_feat: torch.Tensor,
        return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            multimodal_feat: [batch_size, hidden_dim]
            query_feat: [batch_size, hidden_dim]
        """
        batch_size = multimodal_feat.size(0)
        
        # Compute routing weights
        router_input = torch.cat([multimodal_feat, query_feat], dim=-1)
        router_logits = self.router(router_input)
        router_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(router_weights, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Compute weighted expert outputs
        output = torch.zeros(batch_size, self.num_props * 2, 
                            device=multimodal_feat.device, dtype=multimodal_feat.dtype)
        
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [batch_size]
            expert_weights = top_k_weights[:, k:k+1]  # [batch_size, 1]
            
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = multimodal_feat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weights[mask] * expert_output
        
        # Apply sigmoid and reshape
        gauss_params = torch.sigmoid(output).view(batch_size * self.num_props, 2)
        
        return gauss_params, None
