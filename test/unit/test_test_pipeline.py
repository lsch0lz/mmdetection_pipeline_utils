import unittest

from src.test_pipeline import TestPipeline
from src.test_configuration import TestConfiguration


class TestTestPipeline(unittest.TestCase):
    def test_if_test_is_executed_with_given_config_and_model_file(self):
        config_path = "test/data/rpn_r50_fpn.py"
        model_checkpoint = "test/data/resnet50-19c8e357.pth"

        test_pipeline = TestPipeline()
        test_configuration = TestConfiguration(config_path=config_path,
                                               checkpoint_path=model_checkpoint)
        test_pipeline.test(test_configuration)
        assert True
