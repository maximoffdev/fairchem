"""
SO(2) tensor-product convolution adapted for fairchem's tensor-based UMA pipeline.

This mirrors the EBDM SO2_Convolution_TensorProduct behavior while operating on
raw tensors shaped as [num_edges, num_coefficients, channels].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from fairchem.core.models.uma.common.so3 import CoefficientMapping


def _clebsch_gordan_coefficients_array(
    lmax: int, normalize: bool = True
) -> torch.Tensor:
    cg_array = torch.zeros((lmax + 1) ** 2, (lmax + 1) ** 2, (lmax + 1) ** 2)
    for l1 in range(lmax + 1):
        start_idx_1 = l1**2
        length_1 = 2 * l1 + 1
        for l2 in range(lmax + 1):
            start_idx_2 = l2**2
            length_2 = 2 * l2 + 1
            for l3 in range(lmax + 1):
                start_idx_3 = l3**2
                length_3 = 2 * l3 + 1

                if not (abs(l1 - l2) <= l3):
                    continue
                if not (l3 <= (l1 + l2)):
                    continue

                cg_coeff = o3.wigner_3j(l1, l2, l3)
                if normalize:
                    cg_coeff = cg_coeff * math.sqrt(2 * l3 + 1)
                    cg_coeff = cg_coeff * math.sqrt(2 * l2 + 1)

                cg_array[
                    start_idx_1 : (start_idx_1 + length_1),
                    start_idx_2 : (start_idx_2 + length_2),
                    start_idx_3 : (start_idx_3 + length_3),
                ] = cg_coeff

    return torch.permute(cg_array, (2, 0, 1)).contiguous()


def _full_m_index(lmax: int, m: int) -> torch.Tensor:
    # Full l-primary indexing with m in [-l, ..., +l] for each l block.
    indices = []
    for l in range(abs(m), lmax + 1):
        indices.append(l**2 + (m + l))
    return torch.tensor(indices, dtype=torch.long)


def _rescale_tensor_product_weights(tp_weight_data: torch.Tensor, lmax: int) -> None:
    """
    `tp_weight` shape: [L_o, C_o, L_i, C_i, L_f, C_f]
    Rescale in-place to match EBDM tensor-product path normalization.
    """
    irreps = o3.Irreps.spherical_harmonics(lmax, p=1)
    fctp = o3.FullyConnectedTensorProduct(
        irreps,
        irreps,
        irreps,
        irrep_normalization="none",
        path_normalization="element",
    )
    for instruction in fctp.instructions:
        tp_weight_data[
            instruction.i_out,
            :,
            instruction.i_in1,
            :,
            instruction.i_in2,
            :,
        ] *= instruction.path_weight


class SO2_m_Convolution_TensorProduct(nn.Module):
    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()
        assert m >= 0
        assert m <= mmax <= lmax

        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax

        self.weight_start_idx = self.m
        self.weight_length = self.lmax + 1 - self.m

        cg_array = _clebsch_gordan_coefficients_array(self.lmax)

        m_pos_index = _full_m_index(self.lmax, self.m)
        m_0_index = _full_m_index(self.lmax, 0)
        cg_array_m_pos = (
            cg_array.index_select(0, m_pos_index)
            .index_select(1, m_pos_index)
            .index_select(2, m_0_index)
        )
        cg_array_m_pos = cg_array_m_pos.view(
            cg_array_m_pos.shape[0],
            1,
            cg_array_m_pos.shape[1],
            1,
            cg_array_m_pos.shape[2],
            1,
        )
        self.register_buffer("cg_array_m_pos", cg_array_m_pos, persistent=False)

        if self.m > 0:
            m_neg_index = _full_m_index(self.lmax, -self.m)
            cg_array_m_neg = (
                cg_array.index_select(0, m_neg_index)
                .index_select(1, m_pos_index)
                .index_select(2, m_0_index)
            )
            cg_array_m_neg = cg_array_m_neg.view(
                cg_array_m_neg.shape[0],
                1,
                cg_array_m_neg.shape[1],
                1,
                cg_array_m_neg.shape[2],
                1,
            )
            self.register_buffer("cg_array_m_neg", cg_array_m_neg, persistent=False)
        else:
            self.cg_array_m_neg = None

    def forward(
        self,
        x_m: torch.Tensor,
        tp_weight: torch.Tensor,
        tp_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sub_tp_weight = tp_weight[
            self.weight_start_idx : (self.weight_start_idx + self.weight_length),
            :,
            self.weight_start_idx : (self.weight_start_idx + self.weight_length),
            :,
            :,
            :,
        ]

        sub_tp_weight_pos = torch.einsum(
            "ijklmn,ijklmn->ijkln", sub_tp_weight, self.cg_array_m_pos
        )
        sub_tp_weight_pos = sub_tp_weight_pos.reshape(
            sub_tp_weight_pos.shape[0] * sub_tp_weight_pos.shape[1],
            sub_tp_weight_pos.shape[2] * sub_tp_weight_pos.shape[3],
        )

        if self.m > 0:
            sub_tp_weight_neg = torch.einsum(
                "ijklmn,ijklmn->ijkln", sub_tp_weight, self.cg_array_m_neg
            )
            sub_tp_weight_neg = sub_tp_weight_neg.reshape(
                sub_tp_weight_neg.shape[0] * sub_tp_weight_neg.shape[1],
                sub_tp_weight_neg.shape[2] * sub_tp_weight_neg.shape[3],
            )
            weight = torch.cat((sub_tp_weight_pos, sub_tp_weight_neg), dim=0)
            bias = None
        else:
            weight = sub_tp_weight_pos
            if tp_bias is not None:
                zero_tensor = torch.zeros(
                    (weight.shape[0] - self.m_output_channels),
                    device=tp_bias.device,
                    dtype=tp_bias.dtype,
                )
                bias = torch.cat((tp_bias, zero_tensor), dim=0)
            else:
                bias = None

        y_m = F.linear(x_m, weight, bias)

        if self.m > 0:
            num_out_channels = weight.shape[0]
            y_r = y_m.narrow(2, 0, num_out_channels // 2)
            y_i = y_m.narrow(2, num_out_channels // 2, num_out_channels // 2)
            y_m_r = y_r.narrow(1, 0, 1) - y_i.narrow(1, 1, 1)
            y_m_i = y_r.narrow(1, 1, 1) + y_i.narrow(1, 0, 1)
            y_m = torch.cat((y_m_r, y_m_i), dim=1)

        return y_m


class SO2_Convolution_TensorProduct(nn.Module):
    """
    EBDM-style SO(2) tensor-product convolution for fairchem tensor embeddings.
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        internal_weights: bool = True,
        edge_channels_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
    ) -> None:
        super().__init__()
        if not internal_weights:
            raise NotImplementedError(
                "SO2_Convolution_TensorProduct requires internal_weights=True "
                "(matches EBDM behavior)."
            )
        if edge_channels_list is not None:
            raise NotImplementedError(
                "SO2_Convolution_TensorProduct does not use edge_channels_list "
                "(matches EBDM behavior)."
            )

        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.extra_m0_output_channels = extra_m0_output_channels

        self.tp_weight = nn.Parameter(
            torch.randn(
                (self.lmax + 1),
                self.m_output_channels,
                (self.lmax + 1),
                self.sphere_channels,
                (self.lmax + 1),
                1,
            )
        )
        self.tp_weight.data.uniform_(-1.0, 1.0)
        self.tp_weight.data.mul_(1.0 / math.sqrt(self.sphere_channels))
        _rescale_tensor_product_weights(self.tp_weight.data, self.lmax)
        self.tp_bias = nn.Parameter(torch.zeros(self.m_output_channels))

        if self.extra_m0_output_channels is not None:
            self.extra_m0_linear = nn.Linear(
                (self.lmax + 1) * self.sphere_channels,
                self.extra_m0_output_channels,
            )

        self.so2_m_conv = nn.ModuleList()
        for m in range(self.mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Convolution_TensorProduct(
                    m=m,
                    sphere_channels=self.sphere_channels,
                    m_output_channels=self.m_output_channels,
                    lmax=self.lmax,
                    mmax=self.mmax,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,  # kept for API compatibility
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del x_edge
        num_edges = x.shape[0]
        x_m_primary = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)

        out = []
        offset = 0
        x_0_extra = None
        for m in range(self.mmax + 1):
            num_m_components = 1 if m == 0 else 2
            x_m = x_m_primary.narrow(
                1, offset, self.mappingReduced.m_size[m] * num_m_components
            )
            x_m = x_m.reshape(num_edges, num_m_components, -1)
            tp_bias = self.tp_bias if m == 0 else None
            y_m = self.so2_m_conv[m](x_m, self.tp_weight, tp_bias)
            y_m = y_m.view(num_edges, -1, self.m_output_channels)

            if m == 0 and self.extra_m0_output_channels is not None:
                x_0_extra = self.extra_m0_linear(x_m).view(num_edges, -1)

            out.append(y_m)
            offset = offset + self.mappingReduced.m_size[m] * num_m_components

        out_embedding = torch.cat(out, dim=1)
        out_embedding = torch.einsum("nac,ab->nbc", out_embedding, self.mappingReduced.to_m)

        if self.extra_m0_output_channels is not None:
            assert x_0_extra is not None
            return out_embedding, x_0_extra
        return out_embedding
