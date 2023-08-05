from malaya_boilerplate.utils import get_cache_dir
import logging
import os

logger = logging.getLogger(__name__)

try:
    import ctranslate2
except BaseException:
    logger.warning(
        '`ctranslate2` is not available, `use_ctranslate2` is not able to use.')
    ctranslate2 = None


def _check_ctranslate2():
    if ctranslate2 is None:
        raise ModuleNotFoundError(
            'ctranslate2 not installed. Please install it by `pip install ctranslate2` and try again.'
        )


def convert_ctranslate2(model, quantization: str = 'int8',):

    new_path = get_cache_dir(model.replace('/', '-') + f'-{quantization}')
    if not os.path.exists(os.path.join(new_path, 'model.bin')):
        logger.debug('model path is empty, going to convert using ctranslate2.')
        converter = ctranslate2.converters.TransformersConverter(model)
        converter.convert(new_path, quantization=quantization, force=True)

    return new_path


def ctranslate2_translator(
    model,
    quantization: str = 'int8',
    device: str = 'cpu',
    device_index: int = 0,
    **kwargs,
):
    _check_ctranslate2()
    new_path = convert_ctranslate2(model=model, quantization=quantization)
    return ctranslate2.Translator(new_path, device=device, device_index=device_index)


def ctranslate2_generator(
    model,
    quantization: str = 'int8',
    device: str = 'cpu',
    device_index: int = 0,
    **kwargs,
):
    _check_ctranslate2()
    new_path = convert_ctranslate2(model=model, quantization=quantization)
    return ctranslate2.Generator(new_path, device=device, device_index=device_index)
