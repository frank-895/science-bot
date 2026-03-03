# science-bot
An AI-powered agent that analyses life science datasets and answers scientific questions.

## How to use science-bot

1. Save the `capsule_folders.zip` and `data/BixBenchFiltered_50_clean.csv` to your disk. 
2. Run the following commands:

```bash
# to run the benchmark
uv run science-bot benchmark --directory <path to capsule_folders> --csv <path to questions>

# to ask your own question
uv run science-bot run --question <question string> --capsule <name of capsule>
```