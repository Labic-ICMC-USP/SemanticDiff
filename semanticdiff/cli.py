"""
Command-line interface.

Usage:
  semanticdiff run --config semanticdiff.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SemanticDiffConfig
from .pipeline import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="semanticdiff", description="Reflow-resistant PDF semantic diff + LLM report.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run SemanticDiff using a YAML config.")
    run_p.add_argument("--config", required=True, help="Path to YAML config file.")

    args = parser.parse_args()

    if args.cmd == "run":
        cfg = SemanticDiffConfig.from_yaml(args.config)
        run_from_config(cfg)
