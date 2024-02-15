import os
import logging

from mmdet.utils import setup_cache_size_limit_of_dynamo

from mmengine.config import Config
from mmengine.runner import Runner

logger = logging.getLogger(__name__)


class TestPipeline:
    __test__ = False

    def test(self, test_configuration):
        logger.info('Setting up cache size limit of Dynamo')
        setup_cache_size_limit_of_dynamo()

        logger.info(f'Test configuration: {test_configuration}')
        model_config: Config = Config.fromfile(test_configuration.config_path)
        logger.info(f'Model config: {model_config}')

        # if test_configuration.cfg_options is not None:
        #     model_config.merge_from_dict(test_configuration.cfg_options)

        if test_configuration.work_dir is not None:
            model_config.work_dir = test_configuration.work_dir
        elif model_config.get('work_dir', None) is None:
            model_config.work_dir = os.path.join('./work_dirs',
                                                 os.path.splitext(os.path.basename(test_configuration.config_path))[0])

        model_config.load_from = test_configuration.checkpoint_path

        runner = Runner.from_cfg(model_config)

        runner.test()




