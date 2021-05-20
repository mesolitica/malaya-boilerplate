<p align="center">
    <a href="#readme">
        <img alt="logo" width="40%" src="malaya-boilerplate.png">
    </a>
</p>

---

**malaya-boilerplate**, Tensorflow freeze graph optimization and boilerplates to share among Malaya projects.

## Table of contents

  * [malaya_boilerplate.utils](#malaya_boilerplate_utils)
  * [malaya_boilerplate.frozen_graph](#malaya_boilerplate_frozen_graph)

## malaya_boilerplate.utils

#### malaya_boilerplate.utils.available_gpu

```python
def available_gpu():
    """
    Get list of GPUs from `nvidia-smi`.

    Returns
    -------
    result : List[str]
    """
```

#### malaya_boilerplate.utils.gpu_available

```python
def gpu_available():
    """
    Check Malaya is GPU version.

    Returns
    -------
    result : bool
    """
```

#### malaya_boilerplate.utils.print_cache

```python
def print_cache(location = None):
    """
    Print cached data, this will print entire cache folder if let location = None.

    Parameters
    ----------
    location : str, (default=None)
        if location is None, will print entire cache directory.

    """
```

#### malaya_boilerplate.utils.delete_cache

```python
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
```

#### malaya_boilerplate.utils.delete_all_cache

```python
def delete_all_cache():
    """
    Remove cached data, this will delete entire cache folder.
    """
```

#### malaya_boilerplate.utils.close_session

```python
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
```