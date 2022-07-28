import requests
import os
import logging
from tqdm import tqdm
from glob import glob
from .utils import _delete_folder, _get_home

logger = logging.getLogger(__name__)


def check_file_cloud(base_url, url):
    url = base_url + url
    r = requests.head(url)
    exist = r.status_code == 200
    if exist:
        version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
    else:
        version = 0
    return exist, version


def check_local_files(file):
    for key, item in file.items():
        if 'version' in key:
            continue
        if not os.path.isfile(item):
            return False
    return True


def download_file_cloud(base_url, url, filename):
    if 'http' not in url:
        url = base_url + url
    r = requests.get(url, stream=True)
    total_size = int(r.headers['content-length'])
    version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for data in tqdm(
            iterable=r.iter_content(chunk_size=1_048_576),
            total=total_size / 1_048_576,
            unit='MB',
            unit_scale=True,
        ):
            f.write(data)
    return version


def validate_local_file(base_url, url, filename):
    if 'http' not in url:
        url = base_url + url
    r = requests.get(url, stream=True)
    total_size = int(r.headers['content-length'])
    local_size = os.path.getsize(filename)
    validated = local_size == total_size
    if not validated:
        logger.warning(
            f'size of local {filename} ({local_size / 1e6} MB) not matched with size from {url} ({total_size / 1e6} MB)')
    return validated


def download_from_dict(file, s3_file, package, base_url, validate=True, quantized=False):
    home, _ = _get_home(package=package)
    if quantized:
        if 'quantized' not in file:
            f = file['model'].replace(home, '').split('/')
            raise ValueError(
                f'Quantized model for {f[1]} module is not available, please load normal model.'
            )
        model = 'quantized'
        logger.warning('Load quantized model will cause accuracy drop.')
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
                if not os.path.isfile(item) or not validate_local_file(base_url, s3_file[key], item):
                    logger.info(f'downloading frozen {key} to {item}')
                    download_file_cloud(base_url, s3_file[key], item)
            with open(version, 'w') as fopen:
                fopen.write(file['version'])
    else:
        if not check_local_files(file):
            path = file[model]
            path = os.path.sep.join(
                os.path.normpath(path).split(os.path.sep)[1:-1]
            )
            raise OSError(f'{path} is not available, please `validate = True`')


def download_from_string(
    path, module, keys, package, base_url, validate=True, quantized=False
):
    home, _ = _get_home(package=package)
    model = path
    keys = keys.copy()
    keys['version'] = 'version'

    if quantized:
        path = os.path.join(module, f'{path}-quantized')
        quantized_path = os.path.join(path, 'model.pb').replace('\\', '/')
        if not check_file_cloud(base_url, quantized_path)[0]:
            raise ValueError(
                f'Quantized model for `{os.path.join(module, model)}` is not available, please load normal model.'
            )
        logger.warning('Load quantized model will cause accuracy drop.')
    else:
        path = os.path.join(module, path)
    path_local = os.path.join(home, path)
    files_local = {'version': os.path.join(path_local, 'version')}
    files_cloud = {}
    for key, value in keys.items():
        if '/' in value:
            f_local = os.path.join(path_local, value.split('/')[-1])
            f_cloud = value
        else:
            f_local = os.path.join(path_local, value)
            f_cloud = os.path.join(path, value)
            f_cloud = f_cloud.replace('\\', '/')
        files_local[key] = f_local
        files_cloud[key] = f_cloud
    if validate:
        download = False
        version = files_local['version']
        latest = str(
            max(
                [check_file_cloud(base_url, item)[1] for key, item in files_cloud.items()]
            )
        )
        if os.path.isfile(version):
            with open(version) as fopen:
                v = fopen.read()
            if latest not in v:
                p = os.path.dirname(version)
                logger.info(f'Found old version in {p}, deleting..')
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
                if not os.path.isfile(item) or not validate_local_file(base_url, files_cloud[key], item):
                    logger.info(f'downloading frozen {key} to {item}')
                    versions.append(download_file_cloud(base_url, files_cloud[key], item))
            latest = str(max(versions))
            with open(version, 'w') as fopen:
                fopen.write(latest)

    else:
        if not check_local_files(files_local):
            path = files_local['model']
            path = os.path.sep.join(
                os.path.normpath(path).split(os.path.sep)[1:-1]
            )
            raise OSError(f'{path} is not available, please `validate = True`')
    return files_local


def check_file(
    file,
    package,
    base_url,
    s3_file=None,
    module=None,
    keys=None,
    validate=True,
    quantized=False,
    **kwargs,
):
    """
    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized=quantized,
        **kwargs,
    )

    or,

    check_file(path['multinomial'], s3_path['multinomial'], **kwargs)
    """
    if isinstance(file, dict) and isinstance(s3_file, dict):
        download_from_dict(
            file=file,
            s3_file=s3_file,
            package=package,
            base_url=base_url,
            validate=validate,
            quantized=quantized,
        )
    else:
        file = download_from_string(
            path=file,
            module=module,
            keys=keys,
            package=package,
            base_url=base_url,
            validate=validate,
            quantized=quantized,
        )
    return file


def upload(model: str, directory: str, bucket: str = 'malaya',
           application_key_id: str = os.environ.get('backblaze_application_key_id'),
           application_key: str = os.environ.get('backblaze_application_key')):
    """
    Upload directory with malaya-style pattern.

    Parameters
    ----------
    model: str
        it will become directory name.
    directory: str
        local directory with files in it.
    bucket: str, optional (default='malaya')
        backblaze bucket.
    application_key_id: str, optional (default=os.environ.get('backblaze_application_key_id'))
    application_key: str, optional (default=os.environ.get('backblaze_application_key'))
    """

    if not application_key_id or not application_key:
        raise ValueError('must set `backblaze_application_key_id` and `backblaze_application_key` are None.')

    from b2sdk.v1 import B2Api, InMemoryAccountInfo
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)

    b2_api.authorize_account('production', application_key_id, application_key)
    file_info = {'how': 'good-file'}
    b2_bucket = b2_api.get_bucket_by_name(bucket)

    for file in glob(os.path.join(directory, '*')):
        if file.endswith('frozen_model.pb'):
            outPutname = f'{model}/model.pb'
        elif file.endswith('frozen_model.pb.quantized'):
            outPutname = f'{model}-quantized/model.pb'
        else:
            outPutname = f'{model}/{file}'

        b2_bucket.upload_local_file(
            local_file=file,
            file_name=outPutname,
            file_infos=file_info,
        )

        logger.info(f'Uploaded from local {file} to {bucket}/{outPutname}')
