import bz2
import datetime
import json
import time
import yappi
import atexit


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
    shortDateTimeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d___%H-%M-%S')
    return shortDateTimeStamp


def compressFileToBz2(fnmaeIn: str, fnameOut: str, compressionLevel: int=9):
    tarbz2contents = bz2.compress(open(fnmaeIn, 'rb').read(), compressionLevel)
    fh = open(fnameOut, "wb")
    fh.write(tarbz2contents)
    fh.close()


def init_yappi(yappiPath: str, verbose: bool = True):
    '''
    :param yappiPath: where to save yappi's results
    :param verbose: print to console
    :return:
    '''
    OUT_FILE = yappiPath
    if verbose:
        print('[YAPPI START]')
    yappi.set_clock_type('wall')
    yappi.start()

    @atexit.register
    def finish_yappi():
        if verbose:
            print('[YAPPI STOP]')
        yappi.stop()

        if verbose:
            print('[YAPPI WRITE]')

        stats = yappi.get_func_stats()

        for stat_type in ['pstat', 'callgrind', 'ystat']:  # pstat can be read using snakeviz
            print('writing {}.{}'.format(OUT_FILE, stat_type))
            stats.save('{}.{}'.format(OUT_FILE, stat_type), type=stat_type)

        if verbose:
            print('\n[YAPPI FUNC_STATS]')

        print('writing {}.func_stats'.format(OUT_FILE))
        with open('{}.func_stats'.format(OUT_FILE), 'w') as fh:
            stats.print_all(out=fh)

        if verbose:
            print('\n[YAPPI THREAD_STATS]')

        print('writing {}.thread_stats'.format(OUT_FILE))
        tstats = yappi.get_thread_stats()
        with open('{}.thread_stats'.format(OUT_FILE), 'w') as fh:
            tstats.print_all(out=fh)

        if verbose:
            print('[YAPPI OUT]')
