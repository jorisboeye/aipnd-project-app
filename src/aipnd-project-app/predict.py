import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_units", nargs="+", type=int, default=[500, 250])
args = parser.parse_args()

print(args.hidden_units)
