# science-bot
An AI-powered agent that analyses life science datasets and answers scientific questions.

## How to use science-bot

1. Save the `capsule_folders.zip` and `BixBenchFiltered_50_clean.csv` locally. 
2. Clone the repository:

```bash
git clone https://github.com/frank-895/science-bot.git
```

3. Copy `.env.example` and rename to `.env`. Provide an OpenAI API key.
4. To run the benchmark:

```bash
uv run science-bot benchmark --directory <path/to/capsule_folders.zip> --csv <path/to/BixBenchFiltered_50_clean.csv>
```

5. To ask your own question:

```bash
uv run science-bot run --question "question string" --capsule <path/to/capsule>
```

6. Optionally include the argument `--trace-dir <path/to/logs>` to save detailed logs locally.

