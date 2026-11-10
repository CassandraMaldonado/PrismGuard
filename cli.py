# AI evaluations. 

import argparse
import json
from .analysis import analyze

def main():
    parser = argparse.ArgumentParser(description="Evaluate an AI product policy")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--samples", type=int, default=3_000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.seed, args.samples), indent=2))


if __name__ == "__main__":
    main()
