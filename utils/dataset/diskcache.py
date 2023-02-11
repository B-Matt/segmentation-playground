import zlib
import pickle
import diskcache

class DatasetDiskCache(diskcache.Disk):
    """
        Overrides diskcache Disk class with implementation of zlib library for compression.
    """
    def store(self, value, read, key=None):
        """
            Override from base class diskcache.Disk.
            
            :param value: value to convert
            :param bool read: True when value is file-like object
            :return: (size, mode, filename, value) tuple for Cache table
        """

        if read is True:
            value = value.read()
            read = False
        
        value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        value = zlib.compress(value, zlib.Z_BEST_SPEED)
        return super(DatasetDiskCache, self).store(value, read)

    def fetch(self, mode, filename, value, read):
        """
            Override from base class diskcache.Disk.
            :param int mode: value mode raw, binary, text, or pickle
            :param str filename: filename of corresponding value
            :param value: database value
            :param bool read: when True, return an open file handle
            :return: corresponding Python value
        """
        value = super(DatasetDiskCache, self).fetch(mode, filename, value, read)
        if not read:
            value = zlib.decompress(value)
            value = pickle.loads(value)
        return value

def init_cache():
    return diskcache.FanoutCache('.cache',
        disk = DatasetDiskCache,
        shards=8,
        size_limit=50 * (2 ** 30),
        sqlite_mmap_size=2 ** 28,
    )