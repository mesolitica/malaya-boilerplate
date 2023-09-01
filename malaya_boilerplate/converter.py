from malaya_boilerplate.utils import get_cache_dir
import logging
import os

logger = logging.getLogger(__name__)


def _get_ctranslate2():
    try:
        import ctranslate2
        return ctranslate2
    except BaseException:
        raise ModuleNotFoundError(
            'ctranslate2 not installed. Please install it by `pip install ctranslate2` and try again.'
        )


def _get_mlc_llm():
    try:
        from mlc_llm import core as mlc_llm_core
        return mlc_llm_core
    except BaseException:
        raise ModuleNotFoundError(
            'mlc_llm not installed. Please install it by follow https://mlc.ai/mlc-llm/docs/compilation/compile_models.html#install-mlc-llm-package and try again.'
        )


def _get_llama_cpp_python():
    try:
        from llama_cpp import Llama
        return Llama

    except BaseException:
        raise ModuleNotFoundError(
            'llama-cpp-python not installed. Please install it by follow https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal and try again.'
        )


def convert_ctranslate2(model, quantization: str = 'int8'):
    ctranslate2 = _get_ctranslate2()

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
    ctranslate2 = _get_ctranslate2()

    new_path = convert_ctranslate2(model=model, quantization=quantization)
    return ctranslate2.Translator(new_path, device=device, device_index=device_index)


def ctranslate2_generator(
    model,
    quantization: str = 'int8',
    device: str = 'cpu',
    device_index: int = 0,
    **kwargs,
):
    ctranslate2 = _get_ctranslate2()

    new_path = convert_ctranslate2(model=model, quantization=quantization)
    return ctranslate2.Generator(new_path, device=device, device_index=device_index)


def mlc_llm_generator(
    model,
    **kwargs,
):
    mlc_llm_core = _get_mlc_llm()

    args = mlc_llm_core.BuildArgs(
        hf_model=model,
        artifact_path=get_cache_dir(''),
    )
    parsed_args = mlc_llm_core._parse_args(args)
    mlc_llm_core.build_model_from_args(parsed_args)


def llama_cpp_python_generator(
    model,
    **kwargs,
):
    new_path = get_cache_dir(model.replace('/', '-') + '-ggml')
    return Llama(model_path=os.path.join(new_path, ggml-model.bin), **kwargs)
