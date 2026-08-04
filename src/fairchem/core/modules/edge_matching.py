"""
Align edge-level labels with the graph the model actually ran on.

Edge-level labels (e.g. the IQA pairwise subterms ``Vne(A,B)/2``, ``Vee(A,B)/2``, ...)
are stored by the dataset in the order of *its own* ``edge_index`` -- for the IQA pkls
that is the complete directed graph without self-loops, ordered as (u, v) with u in
0..N-1 and, for each u, v in 0..N-1 skipping v == u.

With ``otf_graph: False`` the backbone consumes exactly that ``edge_index``, so the
per-edge predictions line up row-for-row with the labels and nothing has to be done.
With ``otf_graph: True`` the backbone builds a radius graph instead: a differently
ordered *subset* of the dataset edges (and, with ``max_neighbors``, a truncated one).
Comparing those predictions against the label tensor row-by-row would silently pair up
unrelated atom pairs, so the labels have to be re-indexed onto the model's graph first.

This module does that re-indexing. Model edges that have no label (self-loops, or pairs
the dataset does not carry) get a NaN target, which the existing ``torch.isfinite``
output masks already exclude from the loss and the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EdgeAlignment:
    """A mapping from the model's graph edges onto the dataset's edge-label rows.

    Attributes:
        label_index: (E_model,) index into the label rows for each model edge,
            ``-1`` where the model edge has no corresponding label.
        edge_system: (E_model,) index of the system each model edge belongs to.
            Unlike the dataset's edge list, an on-the-fly graph is not guaranteed
            to come out grouped by system, so per-system quantities have to be
            gathered with this rather than ``repeat_interleave``d over counts.
    """

    label_index: torch.Tensor
    edge_system: torch.Tensor

    @property
    def num_edges(self) -> int:
        return self.label_index.numel()

    def nedges(self, num_systems: int) -> torch.Tensor:
        """Per-system model-graph edge counts, the on-the-fly ``batch.nedges``."""
        return torch.bincount(self.edge_system, minlength=num_systems)

    def apply(self, labels: torch.Tensor) -> torch.Tensor:
        """Re-index (E_data, ...) labels onto the model graph -> (E_model, ...).

        Unmatched model edges are filled with NaN so that the ``torch.isfinite``
        based output masks drop them from the loss/metrics.
        """
        matched = self.label_index >= 0
        out = labels.new_full(
            (self.label_index.shape[0], *labels.shape[1:]), float("nan")
        )
        out[matched] = labels[self.label_index[matched]]
        return out


def match_edges_by_node_pairs(
    edge_index_model: torch.Tensor,
    edge_index_data: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Locate each model edge in the dataset's edge list by its (source, target) pair.

    Both edge indices refer to the same (batched) node numbering, so a directed edge
    is uniquely identified by ``source * num_nodes + target``. Returns a
    (E_model,) tensor of row indices into ``edge_index_data``, ``-1`` where the pair
    is absent from the dataset's edge list.

    Note this makes no assumption about how the dataset ordered its edges, which is
    what makes it safe for label graphs that are not complete (e.g. cutoff graphs).
    If the dataset lists a directed pair more than once (periodic images), the match
    is to an arbitrary one of the duplicates.
    """
    keys_data = edge_index_data[0] * num_nodes + edge_index_data[1]
    keys_model = edge_index_model[0] * num_nodes + edge_index_model[1]
    if keys_data.numel() == 0:
        return torch.full_like(keys_model, -1)

    order = torch.argsort(keys_data)
    keys_sorted = keys_data[order]

    pos = torch.searchsorted(keys_sorted, keys_model).clamp(max=keys_sorted.numel() - 1)
    matched = keys_sorted[pos] == keys_model
    return torch.where(matched, order[pos], torch.full_like(pos, -1))


def match_edges_complete_graph(
    edge_index_model: torch.Tensor,
    batch: torch.Tensor,
    natoms: torch.Tensor,
) -> torch.Tensor:
    """Locate each model edge in a *complete directed graph without self-loops*.

    Fallback for when the batch does not carry the dataset's ``edge_index``. Assumes
    the labels of each system are laid out as (u, v) with u in 0..N-1 and, for each u,
    v in 0..N-1 excluding v == u -- i.e. label row ``u * (N - 1) + (v - [v > u])``,
    offset by the label rows of the preceding systems in the batch. Self-loops get
    ``-1``. This is the layout the IQA pkl datasets use.
    """
    natoms = natoms.to(edge_index_model.device)
    # First node and first label row of each system in the batch.
    node_ptr = torch.zeros_like(natoms)
    node_ptr[1:] = torch.cumsum(natoms, dim=0)[:-1]
    labels_per_system = natoms * (natoms - 1)
    label_ptr = torch.zeros_like(labels_per_system)
    label_ptr[1:] = torch.cumsum(labels_per_system, dim=0)[:-1]

    edge_system = batch[edge_index_model[0]]
    local_u = edge_index_model[0] - node_ptr[edge_system]
    local_v = edge_index_model[1] - node_ptr[edge_system]
    n_local = natoms[edge_system]

    # Skip the missing diagonal: v > u shifts the target index down by one.
    offset = local_v - (local_v > local_u).long()
    label_index = label_ptr[edge_system] + local_u * (n_local - 1) + offset
    return torch.where(local_u == local_v, torch.full_like(label_index, -1), label_index)


def build_edge_alignment(
    edge_index_model: torch.Tensor,
    batch: torch.Tensor,
    natoms: torch.Tensor,
    edge_index_data: torch.Tensor | None = None,
) -> EdgeAlignment:
    """Build the model-graph -> label-row mapping for a batch.

    Uses the dataset's own ``edge_index`` when the batch carries it, and otherwise
    falls back to the complete-directed-graph layout.
    """
    if edge_index_data is not None:
        label_index = match_edges_by_node_pairs(
            edge_index_model, edge_index_data, int(batch.numel())
        )
    else:
        label_index = match_edges_complete_graph(edge_index_model, batch, natoms)
    return EdgeAlignment(
        label_index=label_index,
        # An edge never crosses systems, so its source node fixes its system.
        edge_system=batch[edge_index_model[0]],
    )
