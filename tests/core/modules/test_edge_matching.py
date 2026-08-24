"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.modules.edge_matching import (
    build_edge_alignment,
    match_edges_by_node_pairs,
    match_edges_complete_graph,
)


def complete_graph(num_nodes: int, offset: int = 0) -> torch.Tensor:
    """Complete directed graph without self-loops, in the layout the IQA pkls use."""
    u, v = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                u.append(i + offset)
                v.append(j + offset)
    return torch.tensor([u, v])


@pytest.fixture
def batched_graph():
    """Two systems of 3 and 4 atoms, labelled on their complete edge graphs."""
    natoms = torch.tensor([3, 4])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.cat([complete_graph(3, 0), complete_graph(4, 3)], dim=1)
    # each label encodes its own (source, target) pair, so a mismatch is visible
    labels = edge_index[0].float() * 100 + edge_index[1].float()
    return natoms, batch, edge_index, labels


def test_matchers_agree_on_a_reordered_subset(batched_graph):
    natoms, batch, edge_index, _ = batched_graph
    # a radius graph is a permuted subset of the dataset's edges
    subset = torch.tensor([11, 0, 7, 3, 2, 15])
    # ... and self-loops, which no label exists for
    edge_index_model = torch.cat(
        [edge_index[:, subset], torch.tensor([[2, 0], [2, 0]])], dim=1
    )
    expected = torch.cat([subset, torch.tensor([-1, -1])])

    assert torch.equal(
        match_edges_by_node_pairs(edge_index_model, edge_index, batch.numel()), expected
    )
    assert torch.equal(
        match_edges_complete_graph(edge_index_model, batch, natoms), expected
    )


def test_match_by_node_pairs_on_incomplete_label_graph(batched_graph):
    _, batch, edge_index, _ = batched_graph
    # labels defined on a cutoff graph rather than the complete one
    kept = torch.tensor([0, 1, 3, 7, 11, 15])
    edge_index_data = edge_index[:, kept]
    edge_index_model = edge_index[:, torch.tensor([11, 0, 7, 3, 2, 15])]

    label_index = match_edges_by_node_pairs(
        edge_index_model, edge_index_data, batch.numel()
    )
    # edge 2 has no label; the rest map to their row in the reduced list
    assert torch.equal(label_index, torch.tensor([4, 0, 3, 2, -1, 5]))


def test_alignment_reindexes_labels_onto_the_model_graph(batched_graph):
    natoms, batch, edge_index, labels = batched_graph
    edge_index_model = torch.cat(
        [edge_index[:, torch.tensor([11, 0, 7, 3, 2, 15])], torch.tensor([[2], [2]])],
        dim=1,
    )

    alignment = build_edge_alignment(
        edge_index_model, batch, natoms, edge_index_data=edge_index
    )
    target = alignment.apply(labels)

    finite = torch.isfinite(target)
    assert not finite[-1]  # self-loop: no label -> NaN -> masked out of the loss
    assert torch.equal(
        target[finite],
        edge_index_model[0][finite].float() * 100 + edge_index_model[1][finite].float(),
    )
    assert torch.equal(alignment.edge_system, torch.tensor([1, 0, 1, 0, 0, 1, 0]))
    assert torch.equal(alignment.nedges(2), torch.tensor([4, 3]))


def test_alignment_handles_multidimensional_labels(batched_graph):
    natoms, batch, edge_index, labels = batched_graph
    edge_index_model = edge_index[:, torch.tensor([11, 0, 7])]
    vector_labels = torch.stack([labels, -labels], dim=1)

    alignment = build_edge_alignment(
        edge_index_model, batch, natoms, edge_index_data=edge_index
    )
    target = alignment.apply(vector_labels)

    assert target.shape == (3, 2)
    assert torch.equal(target[:, 0], alignment.apply(labels))
    assert torch.equal(target[:, 1], -alignment.apply(labels))


@pytest.mark.parametrize("use_edge_index_data", [True, False])
def test_identical_graphs_are_a_no_op(batched_graph, use_edge_index_data):
    natoms, batch, edge_index, labels = batched_graph
    alignment = build_edge_alignment(
        edge_index,
        batch,
        natoms,
        edge_index_data=edge_index if use_edge_index_data else None,
    )
    assert torch.equal(alignment.apply(labels), labels)
