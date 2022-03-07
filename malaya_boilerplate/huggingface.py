from huggingface_hub import create_repo, upload_file, hf_hub_download
import os
import logging
from glob import glob
from typing import Dict

logger = logging.getLogger('huggingface')

HUGGINGFACE_USERNAME = os.environ.get('HUGGINGFACE_USERNAME', 'huseinzol05')


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
    pass


def upload(model: str, directory: str, username: HUGGINGFACE_USERNAME):
    """
    Upload to huggingface repository, make sure already login using CLI,
    ```
    huggingface-cli login
    ```

    Parameters
    ----------
    model: str
        it will become repository name.
    directory: str
        local directory with files in it.
    username: str, optional (default=os.environ.get('HUGGINGFACE_USERNAME', 'huseinzol05'))
    """
    try:
        create_repo(name=model)
    except Exception as e:
        logger.warning(e)

    repo_id = f'{username}/{model}'

    for file in glob(os.path.join(directory, '*')):
        file_remote = os.path.split(file)[1]
        upload_file(path_or_fileobj=file,
                    path_in_repo=file_remote,
                    repo_id=repo_id)
        logger.info(f'Uploading from {file} to {repo_id}/{file_remote}')


def upload_dict(model: str, files_mapping: Dict[str, str], username: str = HUGGINGFACE_USERNAME):
    """
    Upload to huggingface repository, make sure already login using CLI,
    ```
    huggingface-cli login
    ```

    Parameters
    ----------
    model: str
        it will become repository name.
    files_mapping: Dict[str, str]
        {local_file: target_file}
    username: str, optional (default=os.environ.get('HUGGINGFACE_USERNAME', 'huseinzol05'))
    """
    try:
        create_repo(name=model)
    except Exception as e:
        logger.warning(e)

    repo_id = f'{username}/{model}'

    for file, file_remote in files_mapping.items():
        upload_file(path_or_fileobj=file,
                    path_in_repo=file_remote,
                    repo_id=repo_id)
        logger.info(f'Uploading from {file} to {repo_id}/{file_remote}')
