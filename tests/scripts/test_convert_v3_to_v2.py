# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.lerobot_conversion.convert_v3_to_v2 import convert_data


def _write_data_file(path: Path, episode_indices: list[int], frame_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "episode_index": episode_indices,
            "index": frame_indices,
            "timestamp": [float(index) for index in frame_indices],
        }
    )
    pq.write_table(table, path)


def test_convert_data_uses_global_index_when_file_index_metadata_is_stale(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    converted = tmp_path / "converted"

    _write_data_file(
        source / "data/chunk-000/file-000.parquet",
        episode_indices=[0, 0],
        frame_indices=[0, 1],
    )
    _write_data_file(
        source / "data/chunk-000/file-001.parquet",
        episode_indices=[1, 1, 1],
        frame_indices=[2, 3, 4],
    )

    episode_records = [
        {
            "episode_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": 2,
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
        },
        {
            "episode_index": 1,
            "dataset_from_index": 2,
            "dataset_to_index": 5,
            "length": 3,
            "data/chunk_index": 0,
            # This stale value reproduces LIBERO v3 snapshots where the metadata
            # does not point at the parquet containing the episode rows.
            "data/file_index": 0,
        },
    ]

    convert_data(source, converted, episode_records, chunks_size=1000)

    episode_0 = pq.read_table(converted / "data/chunk-000/episode_000000.parquet")
    episode_1 = pq.read_table(converted / "data/chunk-000/episode_000001.parquet")

    assert episode_0.num_rows == 2
    assert episode_1.num_rows == 3
    assert episode_1.column("index").to_pylist() == [2, 3, 4]
