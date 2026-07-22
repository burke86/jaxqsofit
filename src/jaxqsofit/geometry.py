"""Shared NumPyro geometry helpers for native and embedded spectral fits."""

from __future__ import annotations

import numpy as np

from .model import (
    _line_amplitude_site,
    _line_meta_broad_mask,
    _line_meta_int,
    _ordered_width_site,
)


def line_complex_dense_mass_blocks(tied_line_meta, *, standardized_amplitudes):
    """Return dense blocks for line complexes and the width hierarchy.

    Complexes without ordered broad components retain local amplitude/centroid
    blocks. Complexes with ordered broad components move as complete units
    into the shared width block together with the global broad width and
    unordered width offsets. Sites are assigned to exactly one block.
    """
    blocks = []
    width_complexes = list(tied_line_meta.get("broad_width_order_complex_indices", []))
    width_labels = list(tied_line_meta.get("broad_width_order_site_labels", []))
    centroid_hierarchies = list(tied_line_meta.get("broad_centroid_hierarchy_groups", []))
    ordered_owner_indices = {int(index) for index in width_complexes}
    ordered_complex_sites = {}
    for complex_group in tied_line_meta.get("amp_complex_groups", []):
        complex_index = int(complex_group["complex_index"])
        complex_label = str(complex_group.get("site_label", f"complex_{complex_index}"))
        sites = [_line_amplitude_site(complex_label, standardized=standardized_amplitudes)]
        for hierarchy_index, hierarchy in enumerate(centroid_hierarchies):
            if int(hierarchy.get("complex_index", -1)) == complex_index:
                sites.extend(
                    [
                        f"line_broad_center_{hierarchy_index}_std",
                        f"line_broad_relative_offsets_{hierarchy_index}_std",
                    ]
                )
        if complex_index in ordered_owner_indices:
            ordered_complex_sites[complex_index] = sites
        else:
            blocks.append(tuple(sites))

    unordered_ids = _line_meta_int(tied_line_meta, "unordered_width_group_ids", default=[])
    width_sites = []
    if unordered_ids.size:
        n_w = int(tied_line_meta.get("n_wgroups", 0))
        wgroup = _line_meta_int(tied_line_meta, "wgroup", default=[])
        broad_mask = _line_meta_broad_mask(tied_line_meta)
        if n_w > 0 and wgroup.size == broad_mask.size:
            wgroup_is_broad = np.asarray(
                [np.any(broad_mask[wgroup == gid] > 0.0) for gid in range(n_w)],
                dtype=bool,
            )
            if np.any(wgroup_is_broad[unordered_ids]):
                width_sites.append("line_log_broad_fwhm_std")
        width_sites.append("line_log_fwhm_delta_group_std")
    added_ordered_complexes = set()
    for order_index, owner_index in enumerate(width_complexes):
        owner_index = int(owner_index)
        if owner_index not in added_ordered_complexes:
            width_sites.extend(ordered_complex_sites.get(owner_index, ()))
            added_ordered_complexes.add(owner_index)
        order_label = str(width_labels[order_index]) if order_index < len(width_labels) else str(order_index)
        width_sites.append(_ordered_width_site(order_label, standardized=True))
    if width_sites:
        blocks.append(tuple(width_sites))
    return blocks
