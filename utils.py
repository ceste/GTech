from compress_pickle import dump, load
import io
from hashlib import md5
from time import localtime
import sys
import inspect
import config

def compress_files(filename, obj):
    fname = str(filename)
    dump(obj, fname, compression="lzma", set_default_extension=False)

def load_compressed_files(filename):    
    fname = str(filename)
    return load(fname, compression="lzma", set_default_extension=False)

