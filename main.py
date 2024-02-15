import argparse

from src.pipeline_configuration import PipelineConfiguration
from src.test_configuration import TestConfiguration
from src.test_pipeline import TestPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection test')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the directory to save the file containing evaluation metrics')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline_configuration: PipelineConfiguration = PipelineConfiguration()
    test_configuration: TestConfiguration = TestConfiguration(config_path=args.config)
    test_pipeline: TestPipeline = TestPipeline()

    test_pipeline.test(test_configuration)
