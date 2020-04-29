import re


def normalize_path(path: str):
    """
    :param path: path that may contain backslashes (these are known to cause issues
    :return:
    """
    norm_path = path.replace('\\', '/')
    norm_path = '/'.join(re.split('/', norm_path, maxsplit=0, flags=0)) + '/'
    return norm_path