import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import types_pb2, graph_pb2, attr_value_pb2
from .utils import gpu_available, available_gpu, _get_home

__home__, _ = _get_home()


def check_tf_version():
    version = tf.__version__
    return int(version.split('.')[0])


if check_tf_version() > 1:
    try:
        from tensorflow_addons.utils.resource_loader import LazySO

        _beam_search_so = LazySO('custom_ops/seq2seq/_beam_search_ops.so')
        gather_tree = _beam_search_so.ops.addons_gather_tree
    except:
        import warnings

        warnings.warn(
            'Cannot import beam_search_ops from Tensorflow Addons, `deep_model` for stemmer will not available to use, make sure Tensorflow Addons version >= 0.12.0'
        )


else:
    try:
        from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
    except:
        import warnings

        warnings.warn(
            'Cannot import beam_search_ops from Tensorflow, `deep_model` for stemmer will not available to use, make sure Tensorflow 1 version >= 1.15'
        )


def nodes_session(graph, inputs, outputs, extra = None, attention = None):
    input_nodes = {i: graph.get_tensor_by_name(f'import/{i}:0') for i in inputs}
    output_nodes = {
        o: graph.get_tensor_by_name(f'import/{o}:0') for o in outputs
    }
    if extra:
        extra = {k: graph.get_tensor_by_name(v) for k, v in extra.items()}
        output_nodes = {**output_nodes, **extra}
    if attention:
        output_nodes = {**output_nodes, **attention}
    return input_nodes, output_nodes


def generate_session(graph, **kwargs):
    """
    Load session for a Tensorflow graph.

    Parameters
    ----------
    graph: tensorflow.Graph
    gpu_limit: float, optional (default = 0.999)
        limit percentage to use a gpu memory.

    Returns
    -------
    result : tensorflow.Session
    """
    config = tf.compat.v1.ConfigProto()
    check_gpu = kwargs.get('check_gpu', True)
    if gpu_available() or not check_gpu:
        config.allow_soft_placement = True
        try:
            gpu_limit = float(kwargs.get('gpu_limit', 0.999))
        except:
            raise ValueError('gpu_limit must be a float')
        if not 0 < gpu_limit < 1:
            raise ValueError('gpu_limit must 0 < gpu_limit < 1')

        config.gpu_options.per_process_gpu_memory_fraction = gpu_limit
        config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(config = config, graph = graph)
    return sess


def get_device(**kwargs):
    device = kwargs.get('device', 'CPU:0')
    splitted = device.split(':')
    if len(splitted) != 2:
        raise ValueError('`device` must set as `device:{no}`.')
    if not splitted[1].isdigit():
        raise ValueError('`no` from `device:{no}` is not a digit')
    no = int(splitted[1])
    if no < 0:
        raise ValueError('`no` from `device:{no}` must >= 0')
    device_type = splitted[0].upper()

    if device_type not in ['XLA_CPU', 'XLA_CPU_JIT', 'CPU', 'GPU']:
        raise ValueError(
            "`device` from `device:{no}` must one of ['XLA_CPU', 'XLA_CPU_JIT', 'CPU', 'GPU']"
        )
    check_gpu = kwargs.get('check_gpu', True)
    if gpu_available() or not check_gpu:
        __gpu__ = available_gpu()
        if (not 0 <= no < len(__gpu__)) and check_gpu:
            raise ValueError(f'gpu must 0 <= gpu < {len(__gpu__)}')

    return f'/device:{device}'


def convert_graph_precision(source_graph_def, target_type = 'FP16'):
    def rewrite_batch_norm_node_v2(node, graph_def, target_type = 'FP16'):
        """
        Rewrite FusedBatchNorm with FusedBatchNormV2 for reserve_space_1 and reserve_space_2 in FusedBatchNorm require float32 for 
        gradient calculation (See here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fused-batch-norm)
        """
        if target_type == 'BFLOAT16':
            dtype = types_pb2.DT_BFLOAT16
        elif target_type == 'FP16':
            dtype = types_pb2.DT_HALF
        elif target_type == 'FP64':
            dtype = types_pb2.DT_DOUBLE
        else:
            dtype = types_pb2.DT_FLOAT
        new_node = graph_def.node.add()
        new_node.op = 'FusedBatchNormV2'
        new_node.name = node.name
        new_node.input.extend(node.input)
        new_node.attr['U'].CopyFrom(
            attr_value_pb2.AttrValue(type = types_pb2.DT_FLOAT)
        )
        for attr in list(node.attr.keys()):
            if attr == 'T':
                node.attr[attr].type = dtype
            new_node.attr[attr].CopyFrom(node.attr[attr])

    if target_type == 'BFLOAT16':
        dtype = types_pb2.DT_BFLOAT16
    elif target_type == 'FP16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'FP64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT

    for node in source_graph_def.node:
        if node.op == 'FusedBatchNorm':
            rewrite_batch_norm_node_v2(
                node, target_graph_def, target_type = target_type
            )
            continue
        if ('BatchNorm' in node.name) or ('batch_normalization' in node.name):
            continue
        attrs = list(node.attr.keys())
        if node.op == 'convert_gradient_to_tensor_HBc3xYw22Mw':
            node.op = 'Identity'
            node.attr.setdefault('T')
            node.attr['T'].type = dtype
            del node.attr['_disable_call_shape_inference']

        for attr in attrs:
            if node.attr[attr].type == types_pb2.DT_FLOAT:
                node.attr[attr].type = dtype
            if attr == 'value':
                tensor = node.attr[attr].tensor
                if tensor.dtype == types_pb2.DT_FLOAT:
                    if tensor.float_val:
                        float_val = tf.make_ndarray(node.attr[attr].tensor)
                        node.attr[attr].tensor.CopyFrom(
                            tf.compat.v1.make_tensor_proto(
                                float_val, dtype = dtype
                            )
                        )
                        continue
                    if tensor.tensor_content:
                        tensor_shape = [x.size for x in tensor.tensor_shape.dim]
                        tensor_weights = tf.make_ndarray(tensor)
                        tensor_weights = np.reshape(
                            tensor_weights, tensor_shape
                        )
                        tensor_proto = tf.compat.v1.make_tensor_proto(
                            tensor_weights, dtype = dtype
                        )
                        node.attr[attr].tensor.CopyFrom(tensor_proto)
                        continue

    return source_graph_def


def load_graph(frozen_graph_filename, **kwargs):
    """
    Load frozen graph from a checkpoint.

    Parameters
    ----------
    frozen_graph_filename: str
    use_tensorrt: bool, optional (default=False)
        Use TensorRT.
    tensorrt_precision_mode: str, optional (default='FP32')
        TensorRT precision mode, only supported one of ['FP32', 'FP16', 'INT8'].
        if device is not a gpu, `load_graph` will throw an error.
    precision_mode: str, optional (default='FP32')
        change precision frozen graph, only supported one of ['BFLOAT16', 'FP16', 'FP32', 'FP64'].
    device: str, optional (default='CPU:0')
        device to use for specific model, read more at https://www.tensorflow.org/guide/gpu

    Returns
    -------
    result : tensorflow.Graph
    """

    use_tensorrt = kwargs.get('use_tensorrt', False)
    tensorrt_precision_mode = kwargs.get(
        'tensorrt_precision_mode', 'FP32'
    ).upper()
    precision_mode = kwargs.get('precision_mode', 'FP32').upper()
    device = get_device(**kwargs)

    if tensorrt_precision_mode not in {'FP32', 'FP16', 'INT8'}:
        raise ValueError(
            "`tensorrt_precision_mode` only support one of ['FP32', 'FP16', 'INT8']"
        )

    if precision_mode not in {'BFLOAT16', 'FP16', 'FP32', 'FP64'}:
        raise ValueError(
            "`precision_mode` only support one of ['BFLOAT16', 'FP16', 'FP32', 'FP64']"
        )

    if 'GPU' not in device and use_tensorrt:
        raise ValueError(
            'not able to detect any gpu to use TensorRT, reinstall gpu version and retry.'
        )

    if precision_mode != 'FP32' and use_tensorrt:
        raise ValueError(
            '`precision_mode` must `FP32` if use TensorRT, set `tensorrt_precision_mode` instead.'
        )

    path = frozen_graph_filename.replace(__home__, '')
    path = os.path.sep.join(os.path.normpath(path).split(os.path.sep)[1:-1])

    logging.info(f'running {path} using device {device}')

    with tf.io.gfile.GFile(frozen_graph_filename, 'rb') as f:
        try:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        except Exception as e:
            raise Exception(
                f"{e}, file corrupted due to some reasons, please run `malaya.clear_cache('{path}')` and try again"
            )

    # https://github.com/onnx/tensorflow-onnx/issues/77#issuecomment-445066091
    # to fix import T5
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
            if 'validate_shape' in node.attr:
                del node.attr['validate_shape']
            if len(node.input) == 2:
                node.input[0] = node.input[1]
                del node.input[1]
        elif node.op == 'GatherTree':
            if check_tf_version() > 1:
                node.op = 'Addons>GatherTree'
        elif node.op == 'convert_gradient_to_tensor_HBc3xYw22Mw':
            if use_tensorrt:
                node.op = 'Identity'
                node.attr.setdefault('T')
                del node.attr['_disable_call_shape_inference']

    if use_tensorrt:
        print(
            f'Converting {path} to TensorRT with precision {tensorrt_precision_mode}.'
        )
        try:
            converter = trt.TrtGraphConverter(
                input_graph_def = graph_def,
                precision_mode = tensorrt_precision_mode,
            )
            graph_def = converter.convert()
        except Exception as e:
            raise Exception(
                f'{e}, not able convert {path} to TensorRT with precision {tensorrt_precision_mode}.'
            )

    if precision_mode != 'FP32':
        print(f'Converting {path} to {precision_mode}.')
        try:
            if precision_mode == 'BFLOAT16':
                # some weird error related to range bfloat16
                r = tf.range(0, 10, dtype = tf.bfloat16)
            graph_def = convert_graph_precision(
                graph_def, target_type = precision_mode
            )
        except Exception as e:
            raise Exception(
                f'{e}, not able convert {path} to {precision_mode}.'
            )

    with tf.Graph().as_default() as graph:
        with tf.device(device):
            tf.import_graph_def(graph_def)
    return graph
