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

## Benchmark results

**Accuracy result**: 6% 

| Key       | Value  | Result    |
| --------- | ------ | --------- |
| bix-10-q1 | null   | Incorrect |
| bix-10-q3 | 0.000  | Incorrect |
| bix-11-q2 | null   | Incorrect |
| bix-12-q2 | null   | Incorrect |
| bix-12-q4 | null   | Incorrect |
| bix-13-q2 | null   | Incorrect |
| bix-14-q1 | null   | Incorrect |
| bix-14-q3 | null   | Incorrect |
| bix-16-q2 | null   | Incorrect |
| bix-17-q2 | null   | Incorrect |
| bix-18-q2 | 82000  | Correct   |
| bix-19-q1 | 0.216  | Incorrect |
| bix-19-q3 | 0.956  | Correct   |
| bix-2-q2  | null   | Incorrect |
| bix-20-q2 | null   | Incorrect |
| bix-21-q2 | null   | Incorrect |
| bix-22-q2 | null   | Incorrect |
| bix-25-q1 | null   | Incorrect |
| bix-27-q2 | null   | Incorrect |
| bix-28-q1 | null   | Incorrect |
| bix-28-q3 | null   | Incorrect |
| bix-29-q2 | null   | Incorrect |
| bix-3-q1  | null   | Incorrect |
| bix-3-q3  | null   | Incorrect |
| bix-30-q3 | null   | Incorrect |
| bix-31-q1 | null   | Incorrect |
| bix-31-q3 | null   | Incorrect |
| bix-33-q6 | 36     | Incorrect |
| bix-34-q2 | null   | Incorrect |
| bix-35-q4 | null   | Incorrect |
| bix-36-q3 | null   | Incorrect |
| bix-37-q2 | null   | Incorrect |
| bix-38-q2 | null   | Incorrect |
| bix-4-q1  | null   | Incorrect |
| bix-41-q1 | null   | Incorrect |
| bix-42-q1 | null   | Incorrect |
| bix-43-q3 | null   | Incorrect |
| bix-45-q2 | null   | Incorrect |
| bix-46-q4 | null   | Incorrect |
| bix-47-q3 | NOTCH1 | Correct   |
| bix-51-q1 | null   | Incorrect |
| bix-52-q1 | null   | Incorrect |
| bix-53-q2 | null   | Incorrect |
| bix-54-q2 | null   | Incorrect |
| bix-57-q1 | null   | Incorrect |
| bix-6-q3  | null   | Incorrect |
| bix-61-q4 | null   | Incorrect |
| bix-7-q1  | null   | Incorrect |
| bix-8-q1  | 0.83   | Incorrect |
| bix-9-q3  | null   | Incorrect |

## Write-up

### Approach and Architecture

science-bot is a CLI tool that runs a staged pipeline.
1. Classification
    - The classification stage maps a natural language question into a supported analysis family. Otherwise, it returns unsupported, to quickly identify questions science-bot is not able to answer. 
2. Resolution
    - The resolution stage involves agentic architecture. An iterative loop + scratchpad + tools allows an AI to perform controlled operations. The aim of the resolution stage is to build an execution plan. 
3. Execution
    - The execution stage runs deterministic logic to return a reliable answer, based on the execution plan returned in the resolution stage.

The CLI has two subcommands. One subcommand allows the user to run an arbitrary question against a specified capsule path. The other subcommand runs a predetermined benchmark. When running the benchmark, each question is run as a coroutine (maximum concurrency of 20), so that computaitonal power is not wasted while awaiting LLM responses.

### Trade-offs

- **I avoided heavy preprocessing**, instead inspecting capsules on demand. This kept the system simpler and reduced upfront engineering, but put strain on the resolution stage - which is already the weakset and most complex stage in the pipeline. 
- **I created a general resolver agent rather than specialized agents**. This reduced complexity but also weakened performance. With more time I could have used specialized agents in two ways. Either specializing an agent with the tools and prompts for a specific family or specializing an agent for a particular task, like processing tabular data.
- **A stronger feedback pattern for the agent**. With more time, I would focus on providing the agent structured feedback when it returns a plan that is incomplete or invalid. 

### Failure Analysis

- This was my first attempt at building an AI agent to answer data-based questions. It was a great learning experience and made clear that the hardest part is not the final computation, but getting the agent to reliably find and structure the right data.
- Essentially all the questions in the benchmark fail. This entirely comes down to the resolution stage and the tradeoffs mentioned above - particularly the lack of preprocessing and the lack of specialized agents.
- Sometimes the resolution agent fails due to missing analysis capabilities or unsupported execution types. This is acceptable given time constraints.
- However, the majority of questions are supported. The failure of these questions comes down to limitations with the resolution agent, which often struggles to make sense of the ambiguous or messy data or repeats similar actions without making progress.

### What I Would Improve

If I was doing this task again I would focus on the following improvements:
- Perform really strong preprocessing on the capsules prior to resolution. Summarise the files, extract useful information and metadata. Adding this deterministic layer would reduce computation costs and the number of iterations spent in the resolution stage. 
- Split the general agent into numerous specialized agents. I would experiment with providing each execution type its own agent with a curated set of tools. I would also experiment with building sub-agents for specific tasks - like analysing tabular data or resolving multi-file joins.
- Improve the feedback loop the resolver so that failed searches, invalid plans and dead-end actions consistenctly produce targeted guidance for the next step.
- Add more tools and support more data types. I think the biggest area where I missed easy answers was in the notebooks. Keyword searching tools and precomputed summaries could have improved the agents ability to detect and extract pre-existing answers.
