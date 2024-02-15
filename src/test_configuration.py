class TestConfiguration:
    __test__ = False

    def __init__(self, config_path: str, checkpoint_path: str, work_dir: str = None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.work_dir = work_dir
