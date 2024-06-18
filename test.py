import logging
import sys
from coordinators import OIBenchmarks
from custom import CustomTasks
# from modifiers import SizeOffsetModifier
from runners import FakeBenchmarkRunner


def run():
    results = OIBenchmarks(
        tasks=CustomTasks.from_list([
            {"id": "simple", "prompt": "what is 3 + 4?", "answer": "7"},
            {"id": "hard", "prompt": "who do you think you are??", "answer": "laptop"},
        ]),
        # modifier=SizeOffsetModifier(ntasks=2, offset=0),
        command={},
        runner=FakeBenchmarkRunner(),
        server=False
    ).run()
    # correct_count = sum(1 for result in results if result['status'] == 'correct')
    return results

for _ in range(20):
    print("running!")
    run()
