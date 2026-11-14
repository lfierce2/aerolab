"""Render the workflow figure options: python make_figures.py [A B C]  (default: all)."""
import importlib
import sys

OPTIONS = {"A": "option_a", "B": "option_b", "C": "option_c"}


def main() -> int:
    wanted = [a.upper() for a in sys.argv[1:]] or list(OPTIONS)
    failed = False
    for key in wanted:
        canvas = importlib.import_module(OPTIONS[key]).build()
        failed |= bool(canvas.check())
        stem = canvas.save()
        print(f"  saved {stem}.png and .svg")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
