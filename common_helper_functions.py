import bz2
import datetime
import json
import time


def sort_list(list1, list2) -> list:
    '''
    :param list1:
    :param list2:
    :return: sort list1 by the sorted() ordering of list2.
    Example:    l1 = [3, 2, 1, 4]
                l2 = ['c', 'b', 'a', 'd']
                sort_list(l1, l2)
                >> ['a', 'b', 'c', 'd']
    '''
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


def is_number(s: str):
    # tries to cast string to float. Returns True if s is a number, False otherwise
    try:
        float(s)
        return True
    except ValueError:
        return False


def loadJSONFromFileToDict(jsonPath: str) -> dict:
    json1FileHandler = open(jsonPath)
    json1AsStr = json1FileHandler.read()
    jsonAsDict = json.loads(json1AsStr)

    return jsonAsDict


def generateshortDateTimeStamp(ts: float = None) -> str:
    '''
    :param ts: float, but expects timestamp time.time(). If none, generate timestamp of execution
    :return:
    '''
    if ts is None:
        ts = int(time.time())
    # shortDateTimeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') + '__' + str(ts)
    shortDateTimeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d___%H-%M-%S')
    return shortDateTimeStamp


def compressFileToBz2(fnmaeIn: str, fnameOut: str, compressionLevel: int=9):
    tarbz2contents = bz2.compress(open(fnmaeIn, 'rb').read(), compressionLevel)
    fh = open(fnameOut, "wb")
    fh.write(tarbz2contents)
    fh.close()

