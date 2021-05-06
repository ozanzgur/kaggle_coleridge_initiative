import os

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)