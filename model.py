import torch
import torch.nn.functional as F

class MoE_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Danh sách các expert networks
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        
        # Gate network để kết hợp các expert
        self.gate = torch.nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Tính đầu ra của từng expert: [batch_size, num_experts, output_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Tính gate logits: [batch_size, num_experts]
        gate_logits = self.gate(x)
        # Tính auxiliary loss cho load balancing
        aux_loss = self._compute_load_balancing_loss(gate_logits)

        # Chọn top_k experts dựa trên logits
        topk_logits, topk_indices = torch.topk(gate_logits, self.top_k, dim=1)
        
        # Áp dụng softmax chỉ cho top_k logits đã chọn
        topk_weights = F.softmax(topk_logits, dim=1)  # [batch_size, top_k]
        
        # Chọn outputs của top_k experts
        selected_experts = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        
        # Tổng hợp đầu ra có trọng số từ top_k experts
        topk_weights = topk_weights.unsqueeze(-1)  # [batch_size, top_k, 1]
        output = torch.sum(topk_weights * selected_experts, dim=1)  # [batch_size, output_dim]
        
        return output, aux_loss
    def _compute_load_balancing_loss(self, gate_logits):
        """
        Tính auxiliary loss để cân bằng tải giữa các expert
        Sử dụng 2 thành phần:
        1. Importance loss: đảm bảo mỗi expert được sử dụng tương đương nhau
        2. Load loss: tránh quá tải expert
        """
        # Tính xác suất cho mỗi expert
        gates = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        
        importance_per_expert = gates.mean(dim=0)  # [num_experts]
        # Mục tiêu là phân phối đều cho mỗi expert (1/num_experts)
        target_importance = torch.ones_like(importance_per_expert) / self.num_experts
        importance_loss = F.kl_div(
                importance_per_expert.log(),
                target_importance,
                reduction='sum'
            )
        
        return importance_loss


class MOME_model(torch.nn.Module):
    def __init__(self, env, dataset):
        super(MOME_model, self).__init__()
        self.env = env
        self.n_user = dataset.n_user
        self.m_item = dataset.m_item
        self.free_emb_dimension = self.env.args.free_emb_dimension
        
        # Xử lý dữ liệu đầu vào
        self.feature = torch.tensor(dataset.feature, dtype=torch.float32).to(self.env.device)
        self.feature = torch.nn.functional.normalize(self.feature)
        self.image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        self.image_feat = torch.nn.functional.normalize(self.image_feat)
        self.text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.text_feat = torch.nn.functional.normalize(self.text_feat)

        # User embeddings
        self.user_emb = torch.nn.Embedding(self.n_user, self.free_emb_dimension)
        
        # MoE layers
        self.image_linear = MoE_Layer(self.image_feat.shape[1], self.free_emb_dimension, 4)
        self.text_linear = MoE_Layer(self.text_feat.shape[1], self.free_emb_dimension, 4)
        # Gating network
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(2 * self.free_emb_dimension, self.free_emb_dimension),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.free_emb_dimension, 2)
        )
        
        # Khởi tạo trọng số
        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        self.to(self.env.device)

    def forward(self, random=False):
        user_emb = self.user_emb.weight
        
        # MoE embeddings
        image_emb, image_aux = self.image_linear(self.image_feat)
        text_emb, text_aux = self.text_linear(self.text_feat)
        
        # Kết hợp và tính toán gate
        combined_emb = torch.cat([image_emb, text_emb], dim=1)
        gate_outputs = self.gate(combined_emb)

        # Tính toán gate weights
        gate_weights = torch.softmax(gate_outputs, dim=1)
        
        # Tổng hợp embedding với trọng số
        item_emb = (
            gate_weights[:, 0].unsqueeze(1) * image_emb +
            gate_weights[:, 1].unsqueeze(1) * text_emb
        )
        
        # Add balance loss to encourage equal usage of experts
        mean_gate_weights = gate_weights.mean(dim=0)  # [2]
        target_weights = torch.ones_like(mean_gate_weights) / mean_gate_weights.size(0)
        balance_loss = torch.nn.functional.kl_div(
            mean_gate_weights.log(),
            target_weights,
            reduction='sum'
        )
        return user_emb, item_emb, image_emb, text_emb, image_aux + text_aux, balance_loss