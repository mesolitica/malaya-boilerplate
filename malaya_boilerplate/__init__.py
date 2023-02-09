import sys
import importlib
import logging

logger = logging.getLogger(__name__)

__version__ = '0.0.24rc2'


class Mock:
    def __init__(self, original_name):
        self.original_name = original_name
        self.parent_name = original_name.split('.')[0]
        self.last_path = False
        self.last_name = None

        self.__spec__ = original_name

    def __getattr__(self, name, *args, **kwargs):
        logger.debug(f'self: {self}')
        logger.debug(f'original_name: {self.original_name}')
        logger.debug(f'name: {name}')
        logger.debug(f'last_path: {self.last_path}')
        logger.debug(f'last_name: {self.last_name}')
        if self.last_path:
            self.last_name = name
            mock = Mock(f'{self.original_name}.{name}')
            mock.last_path = True
            return mock
        elif name == '__path__':
            self.last_path = True
            self.last_name = name
        elif name in ['__spec__']:
            self.last_path = False
            self.last_name = None
        elif name in ['v1', 'v2']:
            return
        elif self.last_name == name:
            mock = Mock(f'{self.original_name}.{name}')
            mock.last_path = True
            return mock
        else:
            self.last_path = False
            self.last_name = None
            raise ValueError(f'{self.parent_name} is not installed. Please install it and try again.')

    def __call__(self, *args, **kwargs):
        raise ValueError(f'{self.parent_name} is not installed. Please install it and try again.')


MOCK_MODULES = [
    'tensorflow.compat.v2',
    'tensorflow.compat.v1',
    'tensorflow.signal',
    'tensorflow.core.framework',
    'tensorflow.python.ops',
    'tensorflow.python.framework',
    'tensorflow_probability',
    'tensorflow.keras.preprocessing.sequence',
    'torchlibrosa.stft',
    'tensorflow.python.distribute.cross_device_ops',
    'tensorflow.python.estimator.run_config',
    'tensorflow.python.training',
    'tensorflow.compat.v1.train',
    'tensorflow.python.training.optimizer',
    'tensorflow.compat.v2.io.gfile',
    'tensorflow.python.client',
    'tensorflow',
]
failed = []
for mock in MOCK_MODULES:
    try:
        importlib.import_module(mock)
    except Exception as e:
        logger.debug(f'failed to import {mock}, importing {mock} will replaced with mock module.')
        failed.append(mock)

sys.modules.update((mock, Mock(mock)) for mock in failed)
