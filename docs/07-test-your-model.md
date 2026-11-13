# 7. Test your model

Training loss going down doesn't prove a model got better. Proxmino uses two kinds of test.

## Standard benchmarks

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) runs well-known tests. The
README table uses these, each on the first 200 questions of the test split, with the chat template:

| Test | What it measures |
|---|---|
| GSM8K | School maths word problems (exact final number) |
| IFEval | Following instructions like "answer in 3 bullet points, no commas" |
| ARC-Challenge | Hard school science multiple choice |
| MMLU | General knowledge across 57 subjects |

Always test the plain Granite base the same way, so you can see whether fine-tuning helped or hurt.

## Questions through the app

`training/evals/app_eval.py` asks the running app 35 questions with answers worked out by a program
beforehand: 12 coding tasks checked by running hidden tests (including speed limits), 20 maths and logic
problems graded on the final answer only, 2 questions about an attached spreadsheet, and one UI page checked
for phone layout, dark mode and a working toggle.

```bash
./run.sh &                                   # the app, with your model first in config.yaml
cd training/evals && python app_eval.py my-model
```

Results go to `training/evals/results/my-model.json`, and every answer is saved so you can read why
something failed.

## Be honest about results

- Never train on test questions, and check your data doesn't contain them.
- Report the numbers you measured, including the ones that went down.
- A difference of 1 or 2 points on 200 questions is within noise.
