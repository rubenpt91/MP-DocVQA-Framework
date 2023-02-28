import os
import json
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert eval checkpoint to submission")

    parser.add_argument(
        "-c",
        dest="checkpoint",
        type=str,
        required=True,
        default="save/results/T5_DUDE_concat__2023-02-28_13-16-25.json",
        help="Path to checkpoint with configuration.",
    )
    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_false",
        default=True,
        help="Run DUDEEval directly",
    )
    parser.add_argument(
        "--csv",
        dest="csv",
        action="store_true",
        default=False,
        help="Transform results file to csv for better inspection",
    )
    return parser.parse_args()


def sample_scores_to_submission(r):
    """
    example submission:
    {
        "questionId": "95a82e8934b11e4dca41789f4a70caa3_7b58d9ec29ce6f75d82025df8e6aa91d",
        "answers": ["Moscow Sheremet, Russia - Terminal E - International"],
        "answers_confidence": [0.9298765],  a list with an answer confidence score (1 value), ideally encoded as a 64-bit float between 0 and 1.
        "answers_abstain": False,  a boolean value for flagging documents from an unseen domain, only relevant during evaluation phase 2.
    }
    """
    submission_candidate = []
    for k, values in r["Scores by samples"].items():
        a = {
            "questionId": k,
            "answers": values["pred_answer"].split("|"),
            "answers_confidence": values.get("answers_confidence", -1),
            "answers_abstain": values.get("answers_abstain", False),
        }
        submission_candidate.append(a)
    return submission_candidate


if __name__ == "__main__":
    args = parse_args()

    # TODO: could opt to start from csv as well with updated abstain column
    with open(args.checkpoint, "r") as f:
        predictions = json.load(f)

    if args.csv:
        df = pd.DataFrame.from_dict(predictions["Scores by samples"], orient="index")[
            ["accuracy", "anls", "pred_answer", "answers_confidence"]
        ]
        out_path = args.checkpoint.replace(".json", ".csv")
        df.to_csv(out_path, index=False)
        print(f"Submission file to {out_path}")

    submission_candidate = sample_scores_to_submission(predictions)

    out_path = args.checkpoint.replace("results", "submissions")
    with open(out_path, "w") as f:
        json.dump(submission_candidate, f)
        print(f"Submission file to {out_path}")

    if args.eval:
        GT_PATH = os.path.expanduser(
            "~/code/DUchallenge/annotations/final/DUDE_gt_release-candidate_trainval.json"
        )
        EVAL_SCRIPT = os.path.expanduser("~/code/DUchallenge/DUDEeval/evaluate_submission.py")
        with open(GT_PATH, "r") as f:
            gt = json.load(f)

        gt = [v for v in gt if v["questionId"] in predictions["Scores by samples"]]
        print(f"N : {len(gt)}")
        GT = {"dataset_name": "DUDE Dataset", "dataset_version": "1.0.6", "data": gt}
        gt_out = "/tmp/gt.json"
        with open(gt_out, "w") as f:
            json.dump(GT, f)

        command = f"python3 {EVAL_SCRIPT} -g={gt_out} -s={out_path}"
        os.system(command)
