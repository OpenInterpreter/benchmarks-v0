from benchmark import E2BBenchmarkRunner

runner = E2BBenchmarkRunner()
runner.run(lambda p: None, {}, "sleep for 2 seconds", True)
