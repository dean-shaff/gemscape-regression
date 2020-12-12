import argparse

__all__ = [
    "create_parser"
]

def create_parser(**kwargs) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument("-v", "--verbose", action="store_true", help="Set DEBUG level logging")

    return parser
