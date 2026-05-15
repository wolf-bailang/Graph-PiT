import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.utils import dense_to_sparse, add_self_loops
from torch_scatter import scatter


class HierarchicalAttention(nn.Module):
    """层次化注意力模块，用于处理超级节点和子节点之间的交互"""
    def __init__(self, dim):
        super().__init__()
        # 超级节点到子节点的注意力
        self.super_to_child_attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        # 子节点到超级节点的注意力
        self.child_to_super_attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, super_nodes, child_nodes, super_child_map):
        """
        super_nodes: [n_super, D] - 超级节点特征
        child_nodes: [N, D] - 子节点特征
        super_child_map: [N] - 每个子节点对应的超级节点索引
        """
        n_super, D = super_nodes.shape
        N, _ = child_nodes.shape

        # 1. 超级节点到子节点的消息传递
        # 为每个子节点获取其对应的超级节点特征
        super_feats = super_nodes[super_child_map]  # [N, D]

        # 计算注意力权重
        super_child_cat = torch.cat([super_feats, child_nodes], dim=-1)  # [N, 2D]
        super_to_child_weights = self.super_to_child_attn(super_child_cat).sigmoid()  # [N, 1]

        # 应用注意力权重
        child_nodes_updated = child_nodes + super_to_child_weights * super_feats

        # 2. 子节点到超级节点的消息传递
        # 为每个超级节点，聚合其所有子节点的特征
        child_super_cat = torch.cat([child_nodes, super_feats], dim=-1)  # [N, 2D]
        child_to_super_weights = self.child_to_super_attn(child_super_cat).sigmoid()  # [N, 1]

        # 加权聚合
        weighted_child_feats = child_nodes * child_to_super_weights  # [N, D]

        # 使用scatter_add聚合到对应的超级节点
        super_nodes_updated = torch.zeros_like(super_nodes)
        for i in range(n_super):
            mask = (super_child_map == i)
            if mask.any():
                super_nodes_updated[i] = weighted_child_feats[mask].sum(dim=0) / mask.sum()

        return super_nodes_updated, child_nodes_updated


class HierarchicalGraphLayer(nn.Module):
    """层次化图卷积层，同时处理超级节点图和子图"""
    def __init__(self, dim):
        super().__init__()
        # 超级节点图卷积
        self.super_conv = GATConv(dim, dim, heads=4, concat=False)
        # 子节点图卷积
        self.child_conv = GCNConv(dim, dim)
        # 层次间注意力
        self.hierarchical_attn = HierarchicalAttention(dim)
        # 残差连接和层归一化
        self.layer_norm_super = nn.LayerNorm(dim)
        self.layer_norm_child = nn.LayerNorm(dim)

    def forward(self, super_nodes, child_nodes, super_edge_index, child_edge_index, super_child_map):
        """
        super_nodes: [n_super, D] - 超级节点特征
        child_nodes: [N, D] - 子节点特征
        super_edge_index: [2, E_super] - 超级节点间的边
        child_edge_index: [2, E_child] - 子节点间的边
        super_child_map: [N] - 每个子节点对应的超级节点索引
        """
        # 1. 图内消息传递
        super_nodes_conv = self.super_conv(super_nodes, super_edge_index)
        child_nodes_conv = self.child_conv(child_nodes, child_edge_index)

        # 2. 层次间消息传递
        super_nodes_updated, child_nodes_updated = self.hierarchical_attn(
            super_nodes_conv, child_nodes_conv, super_child_map
        )

        # 3. 残差连接和层归一化
        super_nodes = self.layer_norm_super(super_nodes + super_nodes_updated)
        child_nodes = self.layer_norm_child(child_nodes + child_nodes_updated)

        return super_nodes, child_nodes


class GraphAggregator(nn.Module):
    def __init__(self, dim, num_layers=2, cfg=None):
        super().__init__()
        self.device = torch.device(cfg.device if cfg else "cuda:1")
        self.dim = dim
        self.num_layers = num_layers
        self.lambda_g = cfg.lambda_g
        self.lambda_r = cfg.lambda_r

        # 层次化图卷积层
        self.hierarchical_layers = nn.ModuleList([
            HierarchicalGraphLayer(dim) for _ in range(num_layers)
        ])

        # 超级节点初始化投影
        self.super_node_init = nn.Linear(dim, dim)

        # 边预测MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # 子节点特征增强
        self.child_enhance = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, h, Adj):
        """
        h: [B, N, D] - 图像部件的嵌入向量，例如 [2, 112, 2048]
        Adj: [B, n, n] - 超级节点的链接矩阵，例如 [2, 7, 7]
        返回:
            x_out: [B, N, D] - 处理后的节点特征
            losses: 累计图平滑 + 边预测损失
        """
        # print("h.shape:", h.shape)  # [B, N, D]
        # print("Adj.shape:", Adj.shape)  # [B, n, n]

        h = h.to(self.device)
        Adj = Adj.to(self.device)

        B, total_N, D = h.shape
        n_super = Adj.shape[1]  # 超级节点数
        losses = 0.0
        total_edge_acc = 0.0
        outs = []

        for b in range(B):
            # 1. 计算每个超级节点对应的子节点数
            N = total_N  # 每个样本的节点总数
            if N % n_super != 0:
                # 截断到最大可整除数
                new_N = (N // n_super) * n_super
                h_b = h[b][:new_N, :]  # 截断多余节点
                k = new_N // n_super
            else:
                h_b = h[b]
                k = N // n_super  # 每个超级节点对应的子节点数

            # 2. 构建超级节点图
            # 2.1 初始化超级节点特征 - 使用子节点平均值
            super_nodes = torch.zeros(n_super, D, device=self.device)
            for i in range(n_super):
                start_idx = i * k
                end_idx = (i + 1) * k
                super_nodes[i] = h_b[start_idx:end_idx].mean(dim=0)

            # 应用投影
            super_nodes = self.super_node_init(super_nodes)

            # 2.2 获取超级节点边
            A_super = Adj[b]  # [n_super, n_super]
            if A_super.sum() == 0:
                # 如果没有边，添加自连接
                A_super = torch.eye(n_super, device=self.device)
                # print(f"警告: 样本{b}的超级节点邻接矩阵为空，使用默认自连接")

            super_edge_index, super_edge_attr = dense_to_sparse(A_super)

            # 添加自环确保消息传递
            super_edge_index, _ = add_self_loops(super_edge_index, num_nodes=n_super)

            # 3. 构建子节点图
            child_nodes = h_b  # [N, D]

            # 创建子节点到超级节点的映射
            super_child_map = torch.zeros(child_nodes.size(0), dtype=torch.long, device=self.device)
            for i in range(n_super):
                start_idx = i * k
                end_idx = (i + 1) * k
                super_child_map[start_idx:end_idx] = i

            # 构建子节点之间的边 - 同一超级节点内的子节点全连接
            child_edges_src = []
            child_edges_dst = []

            # 同一超级节点内的子节点全连接
            for i in range(n_super):
                start_idx = i * k
                end_idx = (i + 1) * k
                for j in range(start_idx, end_idx):
                    for l in range(start_idx, end_idx):
                        if j != l:  # 避免自环
                            child_edges_src.append(j)
                            child_edges_dst.append(l)

            child_edge_index = torch.tensor([child_edges_src, child_edges_dst],
                                            device=self.device, dtype=torch.long)

            # 4. 层次化图卷积
            for layer in self.hierarchical_layers:
                super_nodes, child_nodes = layer(
                    super_nodes, child_nodes, super_edge_index,
                    child_edge_index, super_child_map
                )

            # 5. 计算损失
            # 5.1 Laplacian平滑损失 (仅对超级节点)
            if A_super.sum() > 0:
                diff = super_nodes.unsqueeze(1) - super_nodes.unsqueeze(0)  # [n_super, n_super, D]
                squared_diff = diff.pow(2).sum(dim=-1)  # [n_super, n_super]
                L_graph = (squared_diff * A_super).mean()
            else:
                L_graph = torch.tensor(0.0, device=self.device)

            # 5.2 边预测损失
            row, col = super_edge_index
            edge_feat = torch.cat([super_nodes[row], super_nodes[col]], dim=-1)  # [E, 2*D]
            logits = self.edge_mlp(edge_feat)  # [E, 1]

            # 创建目标张量 - 对自环使用0，对原始边使用1
            is_self_loop = row == col
            target = torch.zeros_like(logits)
            for i in range(len(row)):
                if not is_self_loop[i] and A_super[row[i], col[i]] > 0:
                    target[i] = 1.0

            L_rel = F.binary_cross_entropy_with_logits(logits, target)

            # 计算边的预测准确率 (Edge-accuracy)
            with torch.no_grad():
                preds = (logits.squeeze() > 0).float()
                correct = (preds == target.squeeze()).float().sum()
                acc = correct / target.numel() if target.numel() > 0 else 0.0
                total_edge_acc += acc

            # 6. 增强子节点特征
            child_nodes = self.child_enhance(child_nodes)

            # 7. 累加损失
            # print(f"样本 {b} 的损失: L_graph = {L_graph.item()}, L_rel = {L_rel.item()}")
            losses = losses + (self.lambda_g * L_graph + self.lambda_r * L_rel)

            # 8. 收集输出
            outs.append(child_nodes)

        # 拼回 [B, N, D]
        x_out = torch.stack(outs, dim=0)  # [B, N, D]
        
        metrics = {
            "loss": losses / B if B > 0 else 0.0,
            "edge_accuracy": total_edge_acc / B if B > 0 else 0.0
        }

        print(f"平均边预测准确率: {metrics['edge_accuracy'].item()}")

        return x_out, metrics['loss']  # 返回输出和包含损失与准确率的字典