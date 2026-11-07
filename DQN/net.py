import torch
import torch.nn as nn
import torch.nn.functional as F
from env.multiagent.utils import min_max_norm

device = "cuda" if torch.cuda.is_available() else "cpu"


N_ACTIONS = 10
MCS_OBS_SHAPE = 2
IEV_OBS_SHAPE = 2


class MCSNet(nn.Module):
    def __init__(self, ):
        super(MCSNet, self).__init__()
        # self.tmcs = nn.Sequential(
        #     nn.Linear(3, 1),
        #     # nn.Linear(10,1)
        # )
        # self.imcs = nn.Sequential(
        #     nn.Linear(2, 1),
        # )
        # self.qiev = nn.Sequential(
        #     nn.Linear(2, 1),
        # )
        # self.softmax = nn.Softmax(dim=0)

        self.fc1 = nn.Linear(MCS_OBS_SHAPE, 5)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(5, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, obs):
        if len(obs) > 1:
            xT = [[row[col] for row in obs] for col in range(len(obs[0]))]
            xT = [min_max_norm(row) for row in xT]
            x = [[row[col] for row in xT] for col in range(len(xT[0]))]
        else:
            x = obs
        # for m in range(len(obs)):
        #     feat = torch.tensor(obs[m], dtype=torch.float32).to(device)
            # tmcs, imcs, qiev, other = obs[m]['taskMcs'], \
            #                           obs[m]['idleMcs'] ,\
            #                           obs[m]['quasiIev'], \
            #                           torch.tensor(obs[m]['other'], dtype=torch.float32).to(device)
            #
            # tmcs_sum ,imcs_sum, qiev_sum = torch.tensor([0], dtype=torch.float32).to(device), \
            #                                torch.tensor([0], dtype=torch.float32).to(device), \
            #                                torch.tensor([0], dtype=torch.float32).to(device),
            # if len(tmcs) is not 0:
            #     # for i in range(len(tmcs)):
            #     #     tmcs[i] = min_max_norm(tmcs[i])
            #     a = self.tmcs(torch.tensor(tmcs, dtype=torch.float32).to(device)).reshape(1, -1)
            #     tmcs_sum = torch.mean(a)
            #     # a = self.softmax(a)
            #     # for i in range(len(a)):
            #     #     tmcs[i] = torch.tensor([a[0]*j for j in tmcs[i]], dtype=torch.float32, requires_grad=True).to(device)
            #     # for i in range(3):
            #     #     for j in range(len(tmcs)):
            #     #         tmcs_sum[i] += tmcs[j][i]
            #
            # if len(imcs) != 0:
            #     # for i in range(len(imcs)):
            #     #     imcs[i] = min_max_norm(imcs[i])
            #     a = self.imcs(torch.tensor(imcs, dtype=torch.float32).to(device)).reshape(1, -1)
            #     imcs_sum = torch.mean(a)
            #     # a = self.softmax(self.imcs(imcs))
            #     # for i in range(len(a)):
            #     #     imcs[i] = torch.tensor([a[i][0]*j for j in imcs[i]], dtype=torch.float32, requires_grad=True).to(device)
            #     # for i in range(2):
            #     #     for j in range(len(imcs)):
            #     #         imcs_sum[i] += imcs[j][i]
            #
            # if len(qiev) is not 0:
            #     # for i in range(len(qiev)):
            #     #     qiev[i] = min_max_norm(qiev[i])
            #     a = self.qiev(torch.tensor(qiev, dtype=torch.float32).to(device)).reshape(1, -1)
            #     qiev_sum = torch.mean(a)
            #     # a = self.softmax(self.qiev(qiev))
            #     # for i in range(len(a)):
            #     #     qiev[i] = torch.tensor([a[0]*j for j in qiev[i]], dtype=torch.float32, requires_grad=True).to(device)
            #     # for i in range(2):
            #     #     for j in range(len(qiev)):
            #     #         qiev_sum[i] += qiev[j][i]
            # x.append(torch.cat((other.reshape(1,-1), tmcs_sum.reshape(1,-1), imcs_sum.reshape(1,-1), qiev_sum.reshape(1,-1)), dim=1))

        # x = torch.stack(x,0).to(device).squeeze()
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        action_value = self.out(x1)
        return action_value


class IEVNet(nn.Module):
    def __init__(self, ):
        super(IEVNet, self).__init__()

        # self.tmcs = nn.Sequential(
        #     nn.Linear(3, 1),
        # )
        # self.iev = nn.Sequential(
        #     nn.Linear(2, 1),
        # )
        # self.softmax = nn.Softmax(dim=0)

        self.fc1 = nn.Linear(IEV_OBS_SHAPE, 5)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(5, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, obs):
        if len(obs) > 1:
            xT = [[row[col] for row in obs] for col in range(len(obs[0]))]
            xT = [min_max_norm(row) for row in xT]
            x = [[row[col] for row in xT] for col in range(len(xT[0]))]
        else:
            x = obs
        # for m in range(len(obs)):
        #     tmcs, iev, other = obs[m]['taskMcs'], \
        #                               obs[m]['otherIev'] ,\
        #                               torch.tensor(obs[m]['other'], dtype=torch.float32).to(device)
        #
        #     tmcs_sum, iev_sum = torch.tensor([0], dtype=torch.float32).to(device), \
        #                         torch.tensor([0], dtype=torch.float32).to(device)
        #
        #     if len(tmcs) is not 0:
        #         # for i in range(len(tmcs)):
        #         #     tmcs[i] = min_max_norm(tmcs[i])
        #         a = self.tmcs(torch.tensor(tmcs, dtype=torch.float32).to(device)).reshape(1, -1)
        #         tmcs_sum = torch.mean(a)
        #         # a = self.softmax(self.tmcs(tmcs))
        #         # for i in range(len(a)):
        #         #     tmcs[i] = torch.tensor([a[0] * j for j in tmcs[i]], dtype=torch.float32, requires_grad=True).to(device)
        #         # for i in range(3):
        #         #     for j in range(len(tmcs)):
        #         #         tmcs_sum[i] += tmcs[j][i]
        #
        #     if len(iev) is not 0:
        #         # for i in range(len(iev)):
        #         #     iev[i] = min_max_norm(iev[i])
        #         a = self.iev(torch.tensor(iev, dtype=torch.float32).to(device)).reshape(1, -1)
        #         iev_sum = torch.mean(a)
        #         # a = self.softmax(self.iev(iev))
        #         # for i in range(len(a)):
        #         #     iev[i] = torch.tensor([a[0] * j for j in iev[i]], dtype=torch.float32, requires_grad=True).to(device)
        #         # for i in range(2):
        #         #     for j in range(len(iev)):
        #         #         iev_sum[i] += iev[j][i]
        #
        #     x.append(torch.cat((other.reshape(1,-1), tmcs_sum.reshape(1,-1), iev_sum.reshape(1,-1)), dim=1))

        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        action_value = self.out(x1)
        return action_value


def _signed_log_features(values: torch.Tensor) -> torch.Tensor:
    """Compress feature magnitude while preserving a single node's information."""
    return values.sign() * torch.log1p(values.abs())


def _relative_positions(positions: torch.Tensor) -> torch.Tensor:
    """Center coordinates so the graph is translation invariant."""
    centered = positions - positions.mean(dim=0, keepdim=True)
    scale = centered.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
    return centered / scale


def spatial_knn_mask(positions, neighbors: int) -> torch.Tensor:
    """Build a directed spatial k-nearest-neighbor mask with self loops."""
    position_tensor = torch.as_tensor(positions, dtype=torch.float32)
    if position_tensor.ndim != 2 or position_tensor.shape[1] != 2:
        raise ValueError("positions must have shape [number_of_candidates, 2]")
    count = position_tensor.shape[0]
    if count == 0:
        return torch.empty((0, 0), dtype=torch.bool)
    mask = torch.eye(count, dtype=torch.bool, device=position_tensor.device)
    if count == 1 or neighbors <= 0:
        return mask
    distances = torch.cdist(position_tensor, position_tensor)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.argsort(dim=1, stable=True)[:, : min(neighbors, count - 1)]
    mask.scatter_(1, nearest, True)
    return mask


class SpatialGraphQNet(nn.Module):
    """Score variable candidate sets with masked spatial graph attention."""

    def __init__(
        self, feature_dim: int = 2, hidden_dim: int = 32, neighbors: int = 4, heads: int = 4
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.neighbors = neighbors
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.node_encoder = nn.Linear(feature_dim + 2, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, positions):
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        position_tensor = torch.as_tensor(positions, dtype=torch.float32, device=device)
        if feature_tensor.ndim != 2:
            raise ValueError("features must have shape [number_of_candidates, feature_dim]")
        if feature_tensor.shape[0] != position_tensor.shape[0]:
            raise ValueError("features and positions must have the same candidate count")
        if feature_tensor.shape[0] == 0:
            return torch.empty((0, 1), dtype=torch.float32, device=device)
        nodes = torch.cat((_signed_log_features(feature_tensor), _relative_positions(position_tensor)), dim=1)
        hidden = F.relu(self.node_encoder(nodes))
        mask = spatial_knn_mask(position_tensor, self.neighbors).to(device)
        count = hidden.shape[0]
        query = self.query(hidden).reshape(count, self.heads, self.head_dim)
        key = self.key(hidden).reshape(count, self.heads, self.head_dim)
        value = self.value(hidden).reshape(count, self.heads, self.head_dim)
        scores = torch.einsum("ihd,jhd->hij", query, key) / (self.head_dim ** 0.5)
        attention = torch.softmax(scores.masked_fill(~mask.unsqueeze(0), float("-inf")), dim=2)
        messages = torch.einsum("hij,jhd->ihd", attention, value).reshape(count, -1)
        return self.output(F.relu(hidden + messages))


