"""Command-line interface for science-bot."""

import argparse

from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Returns:
        argparse.ArgumentParser: Configured parser for supported arguments.
    """
    parser = argparse.ArgumentParser(prog="science-bot")
    return parser


def main() -> int:
    """Run the top-level science-bot CLI.

    Returns:
        int: Process exit code for the CLI.
    """
    load_dotenv()

    return 1
