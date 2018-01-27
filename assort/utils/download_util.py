import os

from urllib import request


def download(url, fname, directory):
    """Download the file from the given url, filename, and directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    else:
        print("Directory exists: %s" % directory)
    filepath = os.path.join(directory, fname)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (fname, filepath))
        local_fname, _ = request.urlretrieve(url + fname, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded %s bytes %s\n" % (fname, statinfo.st_size))
    else:
        print("File %s exists in %s\n" % (fname, filepath))
    return filepath
