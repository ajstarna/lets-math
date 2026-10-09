"""Command-line entry point for lets-math."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lets-math",
        description="SFT and GRPO experiments for verifiable math problems.",
    )
    parser.add_argument("--version", action="version", version="lets-math 0.1.0")
    return parser


def main() -> None:
    build_parser().parse_args()
