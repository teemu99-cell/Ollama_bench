# ollama-bench

A minimal CLI tool for benchmarking multiple [Ollama](https://ollama.com) models side by side on a set of prompts.

Outputs a comparison table with latency, time to first token (TTFT), generation speed, and a lexical diversity heuristic. Results can optionally be saved as JSON for further analysis.

## Metrics

| Column | Description |
|---|---|
| Latency (s) | Total wall time for the full response (from Ollama's `total_duration`) |
| TTFT (s) | Time to first token: model load time + prompt eval time |
| Tok/s | Generation speed: response tokens / eval duration |
| Resp Tokens | Number of tokens generated |
| Lex Div | Lexical diversity (unique / total tokens). Heuristic only, not a quality judge. |

All timing values come from Ollama's own telemetry fields, not external wall-clock measurement, so they are not affected by network overhead.

## Requirements

- Python 3.9+
- Ollama running locally (or accessible at a known URL)

```
pip install -r requirements.txt
```

## Usage

```bash
# Basic: two models, default prompts file, one run
python ollama_bench.py --models llama3.2 mistral:7b --prompts prompts.txt

# Average over 3 runs per combination
python ollama_bench.py --models llama3.2 mistral:7b --prompts prompts.txt --runs 3

# Remote Ollama instance, save results
python ollama_bench.py --models llama3.2 --prompts prompts.txt --host http://192.168.1.100:11434 --output results.json
```

## Prompt file format

Plain text, one prompt per line. Lines starting with `#` are treated as comments.

```
# My benchmark prompts
Explain the difference between RAG and fine-tuning.
Summarize the key causes of the 2008 financial crisis in three sentences.
```

## Output

```
  [1/4] llama3.2 | prompt 1 | run 1
  [2/4] llama3.2 | prompt 2 | run 1
  [3/4] mistral:7b | prompt 1 | run 1
  [4/4] mistral:7b | prompt 2 | run 1

 Ollama Benchmark Results
+------------+--------+-------------+----------+-------+-------------+---------+
| Model      | Prompt | Latency (s) | TTFT (s) | Tok/s | Resp Tokens | Lex Div |
+============+========+=============+==========+=======+=============+=========+
| llama3.2   | #1     | 4.21        | 0.31     | 58.4  | 221         | 0.74    |
| llama3.2   | #2     | 3.87        | 0.29     | 61.2  | 189         | 0.71    |
| mistral:7b | #1     | 6.44        | 0.48     | 42.1  | 264         | 0.69    |
| mistral:7b | #2     | 5.91        | 0.45     | 44.8  | 237         | 0.67    |
+------------+--------+-------------+----------+-------+-------------+---------+
```

## Notes

- TTFT includes model load time. If the model is already loaded in memory, this will be near zero.
- Lex Div is a surface-level heuristic. It does not measure coherence, factual accuracy, or instruction following. For quality evaluation, feed responses to an LLM judge.
- Run with `--runs 3` or higher to reduce variance, especially for TTFT on cold starts.

## License

MIT
