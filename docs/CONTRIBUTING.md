Benchmarks are an important tool for consistent measuring of how impactful changes are to Open Interpreter.

## Contributing Process

1. Fork the repository and create a new branch for your work.
2. Make changes with clear code comments explaining your approach. Try to follow existing conventions in the code.
3. Open a PR into main linking any related issues. Provide detailed context on your changes.

We will review PRs when possible and work with you to integrate your contribution. Please be patient as reviews take time. Once approved, your code will be merged.

## Add Benchmark

Create a new file for your benchmark (see gaia.py or swe_bench.py for examples)

In `run_benchmarks.py`

- Import your class
- Add new option for `tasks` parameter for instantiating `OIBenchmarks` object
