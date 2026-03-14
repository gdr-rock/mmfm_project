#!/usr/bin/env python3
"""Build a machine-readable COIN taxonomy cache."""

from __future__ import annotations

import argparse

from coin_utils import ensure_taxonomy_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare COIN taxonomy cache")
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument("--taxonomy_xlsx", default="COIN_dataset/taxonomy.xlsx")
    parser.add_argument("--output", default="data/coin/coin_taxonomy.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = ensure_taxonomy_cache(
        args.coin_json,
        args.taxonomy_xlsx,
        args.output,
        force=args.force,
    )
    print(f"Saved taxonomy cache: {args.output}")
    print(f"Tasks: {len(cache['tasks'])}")
    print(f"Step labels: {len(cache['step_id_to_action'])}")


if __name__ == "__main__":
    main()
