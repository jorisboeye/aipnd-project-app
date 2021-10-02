"""
Predict script. Example usage:

python predict.py "./flowers/test/1/image_06743.jpg" "./checkpoints/densenet121/500-250/checkpoint_best.pth" --gpu --top_k 3 --category_names "./flowers/cat_to_name.json"

"""
import argparse
import json

from functions import predict

parser = argparse.ArgumentParser()
parser.add_argument("image_path")
parser.add_argument("checkpoint")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--category_names", type=str, default="")
parser.add_argument(
    "--gpu", dest="device", action="store_const", const="cuda:0", default="cpu"
)

args = parser.parse_args()

if args.category_names:
    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = None

probs, classes = predict(
    image_path=args.image_path,
    checkpoint=args.checkpoint,
    topk=args.top_k,
    device=args.device,
    cat_to_name=cat_to_name,
)

print(probs, classes)
