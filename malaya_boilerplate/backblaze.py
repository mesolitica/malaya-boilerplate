import requests
import os
import logging
from tqdm import tqdm
from .utils import _delete_folder, _get_home
from . import __url__

__home__, _ = _get_home()


def check_file_cloud(url):
    url = __url__ + url
    r = requests.head(url)
    exist = r.status_code == 200
    if exist:
        version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
    else:
        version = 0
    return exist, version


def check_files_local(file):
    for key, item in file.items():
        if 'version' in key:
            continue
        if not os.path.isfile(item):
            return False
    return True


def download_file_cloud(url, filename):
    if 'http' not in url:
        url = __url__ + url
    r = requests.get(url, stream = True)
    total_size = int(r.headers['content-length'])
    version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    with open(filename, 'wb') as f:
        for data in tqdm(
            iterable = r.iter_content(chunk_size = 1_048_576),
            total = total_size / 1_048_576,
            unit = 'MB',
            unit_scale = True,
        ):
            f.write(data)
    return version


def download_from_dict(file, s3_file, validate = True, quantized = False):
    if quantized:
        if 'quantized' not in file:
            f = file.replace(__home__, '').split('/')
            raise ValueError(
                f'Quantized model for {f[1]} module is not available, please load normal model.'
            )
        model = 'quantized'
        logging.warning('Load quantized model will cause accuracy drop.')
    else:
        model = 'model'
    if validate:
        base_location = os.path.dirname(file[model])
        version = os.path.join(base_location, 'version')
        download = False
        if os.path.isfile(version):
            with open(version) as fopen:
                if not file['version'] in fopen.read():
                    print(f'Found old version of {base_location}, deleting..')
                    _delete_folder(base_location)
                    download = True
                else:
                    for key, item in file.items():
                        if not os.path.exists(item):
                            download = True
                            break
        else:
            download = True

        if download:
            for key, item in file.items():
                if 'version' in key:
                    continue
                if model == 'quantized' and key == 'model':
                    continue
                if model == 'model' and key == 'quantized':
                    continue
                if not os.path.isfile(item):
                    print(f'downloading frozen {key} to {item}')
                    download_file_cloud(s3_file[key], item)
            with open(version, 'w') as fopen:
                fopen.write(file['version'])
    else:
        if not check_files_local(file):
            path = file[model]
            path = os.path.sep.join(
                os.path.normpath(path).split(os.path.sep)[1:-1]
            )
            raise Exception(
                f'{path} is not available, please `validate = True`'
            )


def download_from_string(
    path, module, keys, validate = True, quantized = False
):
    model = path
    keys = keys.copy()
    keys['version'] = 'version'

    if quantized:
        path = os.path.join(module, f'{path}-quantized')
        quantized_path = os.path.join(path, 'model.pb')
        if not check_file_cloud(quantized_path)[0]:
            raise Exception(
                f'Quantized model for `{os.path.join(module, model)}` is not available, please load normal model.'
            )
        logging.warning('Load quantized model will cause accuracy drop.')
    else:
        path = os.path.join(module, path)
    path_local = os.path.join(__home__, path)
    files_local = {'version': os.path.join(path_local, 'version')}
    files_cloud = {}
    for key, value in keys.items():
        if '/' in value:
            f_local = os.path.join(path_local, value.split('/')[-1])
            f_cloud = value
        else:
            f_local = os.path.join(path_local, value)
            f_cloud = os.path.join(path, value)
        files_local[key] = f_local
        files_cloud[key] = f_cloud
    if validate:
        download = False
        version = files_local['version']
        latest = str(
            max(
                [check_file_cloud(item)[1] for key, item in files_cloud.items()]
            )
        )
        if os.path.isfile(version):
            with open(version) as fopen:
                if not latest in fopen.read():
                    p = os.path.dirname(version)
                    print(f'Found old version in {p}, deleting..')
                    _delete_folder(p)
                    download = True
                else:
                    for key, item in files_local.items():
                        if not os.path.exists(item):
                            download = True
                            break
        else:
            download = True

        if download:
            versions = []
            for key, item in files_local.items():
                if 'version' in key:
                    continue
                if not os.path.isfile(item):
                    print(f'downloading frozen {key} to {item}')
                    versions.append(download_file_cloud(files_cloud[key], item))
            latest = str(max(versions))
            with open(version, 'w') as fopen:
                fopen.write(latest)

    else:
        if not check_files_local(files_local):
            path = files_local['model']
            path = os.path.sep.join(
                os.path.normpath(path).split(os.path.sep)[1:-1]
            )
            raise Exception(
                f'{path} is not available, please `validate = True`'
            )
    return files_local


def check_file(
    file,
    s3_file = None,
    module = None,
    keys = None,
    validate = True,
    quantized = False,
    **kwargs,
):
    if isinstance(file, dict) and isinstance(s3_file, dict):
        download_from_dict(
            file = file,
            s3_file = s3_file,
            validate = validate,
            quantized = quantized,
        )
    else:
        file = download_from_string(
            path = file,
            module = module,
            keys = keys,
            validate = validate,
            quantized = quantized,
        )
    return file
