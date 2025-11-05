import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm(x1), self.norm(x2), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        q1 = self.to_q(x1)
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)
        qkv2 = self.to_qkv(x2).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv2)

        dots = torch.matmul(q1, k2.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v2)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x1, x2):
        for attn, ff in self.layers:
            x = attn(x1, x2) + x1
            x = ff(x) + x
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (batch_size, num_nodes, in_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, num_nodes, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # shape: (batch_size, num_nodes, num_nodes, 2*out_features)
        return a_input

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=1):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
            for _ in range(nheads)
        ])

        # Output linear layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # Concatenate the outputs of the multi-head attentions
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply the output layer; aggregate with mean as we want a graph-level output
        x = F.elu(self.out_att(x, adj))
        x = torch.mean(x, dim=1)  # Average pooling over nodes to get the graph-level output
        return x

class MLP_improve(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout=0.):
        super(MLP_improve, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiVeh_GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(MultiVeh_GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (batch_size, num_veh, in_features), Wh.shape: (batch_size, num_veh, out_features)
        batch_size, num_veh, _ = Wh.size()

        a_input = torch.cat([Wh.repeat(1, 1, num_veh).view(batch_size, num_veh * num_veh, -1),
                             Wh.repeat(1, num_veh, 1)], dim=2).view(batch_size, num_veh, num_veh, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiVeh_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=1):
        super(MultiVeh_GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            MultiVeh_GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
            for _ in range(nheads)
        ])

        # Output linear layer
        self.out_att = MultiVeh_GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(TrajectoryDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Linear layer to predict future states
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoded_features, future_steps=3):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, encoded_features.size(0), self.hidden_dim).to(encoded_features.device)
        c0 = torch.zeros(self.num_layers, encoded_features.size(0), self.hidden_dim).to(encoded_features.device)

        # Repeat encoded features for each future step
        lstm_input = encoded_features.unsqueeze(1).repeat(1, future_steps, 1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))

        # Predict future states
        output = self.fc(lstm_out)

        return output

# new network layers
class GAT_attpool(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=1):
        super(GAT_attpool, self).__init__()
        self.dropout = dropout
        self.nheads = nheads

        # 多头注意力层
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
            for _ in range(nheads)
        ])

        # 输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

        # **注意力池化**
        self.attention_vector = nn.Parameter(torch.empty(size=(nclass, 1)))
        nn.init.xavier_uniform_(self.attention_vector.data, gain=1.414)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        # **多头注意力计算**
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # (batch_size, num_nodes, nhid * nheads)
        x = F.dropout(x, self.dropout, training=self.training)

        # **通过输出层进一步计算特征**
        x = F.elu(self.out_att(x, adj))  # (batch_size, num_nodes, nclass)

        # **注意力池化**
        att_weights = F.softmax(torch.matmul(x, self.attention_vector), dim=1)  # (batch_size, num_nodes, 1)
        x = torch.sum(att_weights * x, dim=1)  # (batch_size, nclass)

        return x


class TemporalCNN(nn.Module):
    """时序特征提取模块-使用1D CNN"""

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim[0], kernel_size=5, padding=2,
                               stride=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim[0], out_channels=hidden_dim[1], kernel_size=5, padding=2,
                               stride=1)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.hidden_dim = hidden_dim[1]

    def forward(self, x):
        # x shape: (batch_size, seq_len=5, input_dim=5)
        # 转换维度以适应1D卷积
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)

        # 两层卷积+激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 最大池化并展平
        x = self.pool(x)
        x = x.view(-1, self.hidden_dim)
        return x


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        batch_size = h.size(0)
        N = h.size(1)

        Wh = torch.matmul(h, self.W)
        Wh = Wh.view(batch_size, N, self.num_heads, self.out_dim)

        a_input = torch.cat([Wh.repeat(1, 1, 1, N).view(batch_size, N * self.num_heads, N, self.out_dim),
                             Wh.repeat(1, N, 1, 1).view(batch_size, N * self.num_heads, N, self.out_dim)], dim=3)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        adj = adj.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        e = e.view(batch_size, self.num_heads, N, N)
        attention = F.softmax(e.masked_fill(adj == 0, float('-inf')), dim=3)

        Wh = Wh.transpose(1, 2)
        h_prime = torch.matmul(attention, Wh)

        h_prime = h_prime.transpose(1, 2).contiguous()
        h_prime = h_prime.view(batch_size, N, -1)

        return h_prime


class BackwardCausalReasoning(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=96):
        super().__init__()

        # 前向影响编码器
        self.forward_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 处理拼接后的特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 注意力层
        self.query_transform = nn.Linear(input_dim, hidden_dim)
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, h_front, h_ego, h_back):
        """
        Args:
            h_front: (batch_size, num_vehicles, input_dim)
            h_ego: (batch_size, 1, input_dim)
            h_back: (batch_size, num_vehicles, input_dim)
        """
        batch_size = h_front.size(0)
        num_vehicles = h_front.size(1)

        # 将h_ego扩展以匹配每个前车
        h_ego_expanded = h_ego.expand(-1, num_vehicles, -1)  # (batch_size, num_vehicles, input_dim)

        # 连接前车特征和自车特征
        front_ego_concat = torch.cat([h_front, h_ego_expanded], dim=-1)  # (batch_size, num_vehicles, input_dim*2)

        # 对每个车辆进行编码
        forward_encoding = self.forward_encoder(front_ego_concat)  # (batch_size, num_vehicles, hidden_dim)

        # 转换后车特征为查询向量
        Q = self.query_transform(h_back)  # (batch_size, num_vehicles, hidden_dim)
        K = self.key_transform(forward_encoding)  # (batch_size, num_vehicles, hidden_dim)
        V = self.value_transform(forward_encoding)  # (batch_size, num_vehicles, hidden_dim)

        # 计算注意力分数
        attention = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_vehicles, num_vehicles)
        attention = attention / (K.size(-1) ** 0.5)
        attention = F.softmax(attention, dim=-1)

        # 聚合特征
        context = torch.matmul(attention, V)  # (batch_size, num_vehicles, hidden_dim)

        # 对所有车辆的特征进行平均
        context = context.mean(dim=1)  # (batch_size, hidden_dim)
        Q = Q.mean(dim=1)  # (batch_size, hidden_dim)

        # 连接并输出
        output = torch.cat([context, Q], dim=1)  # (batch_size, hidden_dim*2)
        return self.output_layer(output)  # (batch_size, output_dim)


class MultiLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, activation=nn.ReLU, dropout_rate=0.0):
        """
        通用多层 MLP 网络

        参数:
            input_dim (int): 输入张量的维度
            output_dim (int): 输出张量的维度
            hidden_dims (list): 每个隐藏层的维度大小的列表，列表长度决定隐藏层数量
                               如果为 None，默认为一个隐藏层 [64]
            activation (nn.Module): 激活函数，默认为 ReLU
            dropout_rate (float): dropout 概率，默认为 0 (不使用 dropout)
        """
        super(MultiLayerMLP, self).__init__()

        # 如果 hidden_dims 为 None，默认使用一个隐藏层
        if hidden_dims is None:
            hidden_dims = [64]

        # 创建层列表
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # 添加其他隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # 将所有层组合成一个序列模型
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        参数:
            x (tensor): 输入张量，最后一维需要等于 input_dim

        返回:
            tensor: 输出张量，最后一维等于 output_dim
        """
        return self.mlp(x)