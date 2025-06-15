import os
import json
import torch
import argparse
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=("MATH", "WildChat10K", "DS-1000", "MMLU"))
parser.add_argument("--tree_path", type=str, required=True)
parser.add_argument("--results_path", type=str, required=True)
args = parser.parse_args()

print(f"[INFO] Loading capability tree from: Datasets/{args.dataset}/EvalTree/{args.tree_path}.json")
with open(os.path.join("".format(args.dataset), "{}.json".format(args.tree_path)), "r") as f:
    TREE = json.load(f)

print(f"[INFO] Loading evaluation results from: Datasets/{args.dataset}/eval_results/{args.results_path}/results.json")
with open( f"{args.results_path}.json", "r") as fin:
    RESULTS = json.load(fin)

valid_count = sum(1 for v in RESULTS if v != -1)
print(f"[INFO] Valid results count = {valid_count} out of {len(RESULTS)}")


# Determine the type of metric
if args.dataset in ("MATH", "DS-1000", "MMLU"):
    results_type = "accuracy"
elif args.dataset == "WildChat10K":
    results_type = "win-rate"
else:
    raise NotImplementedError("dataset = {}".format(args.dataset))
# print(f"[INFO] Evaluation metric type: {results_type}")


def calculate(tree):
    if not isinstance(tree, int):
        tree_results = {
            "size": 0,
            "sum_metrics": 0,
        }

        if isinstance(tree["subtrees"], list):
            assert ("kmeans" not in tree) or (tree["kmeans"] is None)

            tree_results["subtrees"] = []
            for subtree in tree["subtrees"]:
                subtree_results = calculate(subtree)
                tree_results["subtrees"].append(subtree_results)
                tree_results["size"] += subtree_results["size"]
                tree_results["sum_metrics"] += subtree_results["sum_metrics"]

        elif isinstance(tree["subtrees"], dict):
            assert "kmeans" in tree and tree["kmeans"] is not None

            tree_results["subtrees"] = {}
            for cluster, subtree in tree["subtrees"].items():
                subtree_results = calculate(subtree)
                tree_results["subtrees"][cluster] = subtree_results
                tree_results["size"] += subtree_results["size"]
                tree_results["sum_metrics"] += subtree_results["sum_metrics"]

        else:
            # case: subtrees is a single leaf node (int)
            subtree_results = calculate(tree["subtrees"])
            tree_results["size"] += subtree_results["size"]
            tree_results["sum_metrics"] += subtree_results["sum_metrics"]
            tree_results["subtrees"] = subtree_results

    else:
        metrics = RESULTS[tree]
        if results_type == "accuracy":
            assert metrics in (0, 1)
            tree_results = {
                "size": 1,
                "sum_metrics": metrics,
            }
        elif results_type == "win-rate":
            assert isinstance(metrics, list) and len(metrics) == 2
            assert metrics[0] in (1, 2) and metrics[1] in (1, 2)
            tree_results = {
                "size": 1,
                "sum_metrics": int(metrics[0] == 1) + int(metrics[1] == 1),
            }
        else:
            raise NotImplementedError("results_type = {}".format(results_type))
        tree_results["subtrees"] = tree

    # --- CI ---
    if tree_results["size"] < 3:
        tree_results["confidence_interval"] = None
    else:
        tree_results["confidence_interval"] = {}
        for alpha in (0.01, 0.05):
            if results_type == "accuracy":
                lower_bound, upper_bound = sm.stats.proportion_confint(
                    tree_results["sum_metrics"], tree_results["size"],
                    alpha=alpha, method="beta"
                )
            elif results_type == "win-rate":
                lower_bound, upper_bound = sm.stats.proportion_confint(
                    tree_results["sum_metrics"], tree_results["size"] * 2,
                    alpha=alpha, method="beta"
                )
            else:
                raise NotImplementedError("results_type = {}".format(results_type))
            tree_results["confidence_interval"][alpha] = (lower_bound, upper_bound)

    return tree_results

print("[INFO] Starting tree traversal and CI calculation...")
TREE_RESULTS = calculate(TREE)

# Save output
print("[INFO] Saving output...")
# output_path = args.tree_path.split("/")
# assert len(output_path) == 2
output_file = f"confidence_interval_DOVE[dataset={args.dataset}]_[tree={args.tree_path}]_[results={args.results_path}].json"

# שמירה לתיקייה הנוכחית
with open(output_file, "w") as fout:
    json.dump(TREE_RESULTS, fout, indent=2)

print(f"[DONE] Output saved to: {output_file}")
