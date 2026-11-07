import torch

from DQN.net import SpatialGraphQNet, spatial_knn_mask


def test_knn_mask_keeps_self_and_nearest_neighbor():
    positions = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]])

    mask = spatial_knn_mask(positions, neighbors=1)

    assert mask.dtype is torch.bool
    assert mask.tolist() == [[True, True, False], [True, True, False], [False, True, True]]


def test_spatial_graph_q_net_returns_one_value_per_candidate():
    network = SpatialGraphQNet(hidden_dim=8, neighbors=2)
    features = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    positions = [[103.9, 30.6], [104.0, 30.65], [104.1, 30.7]]

    values = network(features, positions)

    assert values.shape == (3, 1)
    assert torch.isfinite(values).all()


def test_spatial_graph_q_net_handles_a_single_candidate():
    network = SpatialGraphQNet(hidden_dim=8, neighbors=4)

    first = network([[1.0, 2.0]], [[103.9, 30.6]])
    second = network([[9.0, 5.0]], [[103.9, 30.6]])

    assert first.shape == (1, 1)
    assert not torch.isclose(first, second)


def test_neighbor_features_change_connected_candidate_value():
    connected = SpatialGraphQNet(hidden_dim=4, neighbors=1, heads=2)
    isolated = SpatialGraphQNet(hidden_dim=4, neighbors=0, heads=2)
    for parameter in connected.parameters():
        torch.nn.init.constant_(parameter, 0.1)
    isolated.load_state_dict(connected.state_dict())
    positions = [[0.0, 0.0], [0.0, 1.0], [10.0, 0.0]]

    baseline = connected([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], positions)
    changed = connected([[1.0, 1.0], [10.0, 10.0], [0.0, 0.0]], positions)
    isolated_baseline = isolated([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], positions)
    isolated_changed = isolated([[1.0, 1.0], [10.0, 10.0], [0.0, 0.0]], positions)

    assert not torch.isclose(baseline[0], changed[0])
    assert torch.isclose(isolated_baseline[0], isolated_changed[0])


def test_graph_net_uses_multiple_attention_heads():
    network = SpatialGraphQNet(hidden_dim=8, heads=2)

    assert network.heads == 2
    assert network.head_dim == 4


def test_knn_mask_breaks_equal_distance_ties_by_input_order():
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

    mask = spatial_knn_mask(positions, neighbors=1)

    assert mask[0].tolist() == [True, True, False]
