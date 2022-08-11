from shutil import rmtree
from pathlib import Path
from packaging import version
import tensorflow as tf
import logging
import os

logger = logging.getLogger(__name__)
DEVICES = None


def _delete_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))


def _get_home(package):
    home = os.environ.get(
        f'{package.upper()}_CACHE',
        os.path.join(str(Path.home()), package.title()),
    )
    version_path = os.path.join(home, 'version')
    return home, version_path


def _delete_macos(package):
    home, _ = _get_home(package)
    macos = os.path.join(home, '__MACOSX')
    if os.path.exists(macos):
        rmtree(macos)


def get_home(package, package_version):
    home, version_path = _get_home(package)

    try:
        if not os.path.exists(home):
            os.makedirs(home)
    except BaseException:
        raise Exception(
            f'Malaya cannot make directory for caching. Please check your {home}'
        )

    _delete_macos(package=package)
    if not os.path.isfile(version_path):
        with open(version_path, 'w') as fopen:
            fopen.write(package_version)
    else:
        with open(version_path, 'r') as fopen:
            cached_version = fopen.read()
        try:
            if float(cached_version) < 1:
                _delete_folder(home)
                with open(version_path, 'w') as fopen:
                    fopen.write(package_version)
        except BaseException:
            with open(version_path, 'w') as fopen:
                fopen.write(package_version)

    return home, version_path


def describe_availability(dict, transpose=True, text=''):
    if len(text):
        logger.info(text)
    try:
        import pandas as pd

        df = pd.DataFrame(dict)

        if transpose:
            return df.T
        else:
            return df
    except BaseException:
        return dict


def available_device(refresh=False):
    """
    Get list of devices and memory limit from `tensorflow.python.client.device_lib.list_local_devices()`.

    Returns
    -------
    result : List[str]
    """
    global DEVICES

    if DEVICES is None and not refresh:
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


def available_gpu(refresh=False):
    """
    Get list of GPUs and memory limit from `tensorflow.python.client.device_lib.list_local_devices()`.

    Returns
    -------
    result : List[str]
    """

    devices = available_device(refresh=refresh)
    return [d for d in devices if 'GPU' in d[0] and 'XLA' not in d[0]]


def print_cache(package, location=None):
    """
    Print cached data, this will print entire cache folder if let location = None.

    Parameters
    ----------
    location : str, (default=None)
        if location is None, will print entire cache directory.

    """

    home, _ = _get_home(package=package)
    path = os.path.join(home, location) if location else home
    paths = DisplayablePath.make_tree(Path(path))
    for path in paths:
        print(path.displayable())


def delete_cache(package, location):
    """
    Remove selected cached data, please run print_cache() to get path.

    Parameters
    ----------
    location : str

    Returns
    -------
    result : boolean
    """

    home, _ = _get_home(package=package)
    if not isinstance(location, str):
        raise ValueError('location must be a string')
    location = os.path.join(home, location)
    if not os.path.exists(location):
        raise Exception(
            f"folder not exist, please check path from `{package.replace('-', '_')}.utils.print_cache()`"
        )
    if not os.path.isdir(location):
        raise Exception(
            f"Please use parent directory, please check path from `{package.replace('-', '_')}.utils.print_cache()`"
        )
    _delete_folder(location)
    return True


def delete_all_cache(package):
    """
    Remove cached data, this will delete entire cache folder.
    """
    _delete_macos(package)
    home, _ = _get_home(package)
    try:
        _delete_folder(home)
        with open(version_path, 'w') as fopen:
            fopen.write(version)
        return True
    except BaseException:
        raise Exception(
            f'failed to clear cached models. Please make sure {home} is able to overwrite from {package}'
        )


def close_session(model):
    """
    Close session from a model to prevent any out-of-memory or segmentation fault issues.

    Parameters
    ----------
    model : Malaya object.

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
        logger.warning(e)
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
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria
        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path,
                    parent=displayable_root,
                    is_last=is_last,
                    criteria=criteria,
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


def check_tf2_huggingface():
    if version.parse(tf.__version__) < version.parse('2.0'):
        raise Exception('Tensorflow version must >= 2.0 to use HuggingFace models.')


def check_tf2(func):

    def inner1(*args, **kwargs):

        check_tf2_huggingface()
        return func(*args, **kwargs)

    return inner1
