import os
import benchmark

runner = benchmark.DockerBenchmarkRunner()
messages = runner.run(lambda _: None, {"auto_run": True, "api_key": os.environ.get("OPENAI_API_KEY", "")}, "sleep for 2 seconds", True)
print(messages)