from shutil import rmtree
from pathlib import Path
import os
import logging
from . import __package__, __package_version__

IS_GPU = None
DEVICES = None


def _delete_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))


def _get_home():
    home = os.environ.get(
        f'{__package__.upper()}_CACHE',
        os.path.join(str(Path.home()), __package__.title()),
    )
    version_path = os.path.join(home, 'version')
    return home, version_path


def _delete_macos():
    home, _ = _get_home()
    macos = os.path.join(home, '__MACOSX')
    if os.path.exists(macos):
        rmtree(macos)


def get_home():
    home, version_path = _get_home()

    try:
        if not os.path.exists(home):
            os.makedirs(home)
    except:
        raise Exception(
            f'Malaya cannot make directory for caching. Please check your {home}'
        )

    _delete_macos()
    if not os.path.isfile(version_path):
        with open(version_path, 'w') as fopen:
            fopen.write(__package_version__)
    else:
        with open(version_path, 'r') as fopen:
            cached_version = fopen.read()
        try:
            if float(cached_version) < 1:
                _delete_folder(home)
                with open(version_path, 'w') as fopen:
                    fopen.write(__package_version__)
        except:
            with open(version_path, 'w') as fopen:
                fopen.write(__package_version__)

    return home, version_path


def describe_availability(dict, transpose = True, text = ''):
    if len(text):
        logging.basicConfig(level = logging.INFO)

        logging.info(text)
    try:
        import pandas as pd

        df = pd.DataFrame(dict)

        if transpose:
            return df.T
        else:
            return df
    except:
        return dict


def available_device():
    """
    Get list of devices and memory limit from `tensorflow.python.client.device_lib.list_local_devices()`.

    Returns
    -------
    result : List[str]
    """
    global DEVICES

    if DEVICES is None:
        from tensorflow.python.client import device_lib

        DEVICES = device_lib.list_local_devices()
        DEVICES = [
            (
                i.name.replace('/device:', ''),
                f'{round(i.memory_limit / 1e9, 3)} GB',
            )
            for i in DEVICES
        ]

    return DEVICES


def available_gpu():
    """
    Get list of GPUs and memory limit from `tensorflow.python.client.device_lib.list_local_devices()`.

    Returns
    -------
    result : List[str]
    """

    devices = available_device()
    return [d for d in devices if 'GPU' in d[0] and 'XLA' not in d[0]]


def gpu_available():
    """
    Check Malaya is GPU version.

    Returns
    -------
    result : bool
    """

    import pkg_resources

    global IS_GPU

    if IS_GPU is None:
        IS_GPU = f'{__package__}-gpu' in [
            p.project_name for p in pkg_resources.working_set
        ]
        if IS_GPU:
            gpus = available_gpu()
            IS_GPU = len(gpus) > 0
    return IS_GPU


def is_gpu_version():
    """
    Check Malaya is GPU version.

    Returns
    -------
    result : bool
    """
    return gpu_available()


def print_cache(location = None):
    """
    Print cached data, this will print entire cache folder if let location = None.

    Parameters
    ----------
    location : str, (default=None)
        if location is None, will print entire cache directory.

    """

    home, _ = _get_home()
    path = os.path.join(home, location) if location else home
    paths = DisplayablePath.make_tree(Path(path))
    for path in paths:
        print(path.displayable())


def delete_cache(location):
    """
    Remove selected cached data, please run print_cache() to get path.

    Parameters
    ----------
    location : str

    Returns
    -------
    result : boolean
    """

    home, _ = _get_home()
    if not isinstance(location, str):
        raise ValueError('location must be a string')
    location = os.path.join(home, location)
    if not os.path.exists(location):
        raise Exception(
            f'folder not exist, please check path from `{__package__}.utils.print_cache()`'
        )
    if not os.path.isdir(location):
        raise Exception(
            f'Please use parent directory, please check path from `{__package__}.utils.print_cache()`'
        )
    _delete_folder(location)
    return True


def delete_all_cache():
    """
    Remove cached data, this will delete entire cache folder.
    """
    _delete_macos()
    home, _ = _get_home()
    try:
        _delete_folder(home)
        with open(version_path, 'w') as fopen:
            fopen.write(version)
        return True
    except:
        raise Exception(
            f'failed to clear cached models. Please make sure {home} is able to overwrite from Malaya'
        )


def close_session(model):
    """
    Close session from a model to prevent any out-of-memory or segmentation fault issues.

    Parameters
    ----------
    model : malaya object.

    Returns
    -------
    result : boolean
    """

    success = False
    try:
        if hasattr(model, 'sess'):
            model.sess.close()
            success = True
        elif hasattr(model, '_sess'):
            model._sess.close()
            success = True
    except Exception as e:
        logging.warning(e)
    return success


class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + os.path.sep
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent = None, is_last = False, criteria = None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria
        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key = lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path,
                    parent = displayable_root,
                    is_last = is_last,
                    criteria = criteria,
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + os.path.sep
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ['{!s} {!s}'.format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return ''.join(reversed(parts))
