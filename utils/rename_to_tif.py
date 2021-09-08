import os

def rttif(mypath):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        for dirname in dirnames:
            folder = os.path.join(dirpath, dirname)
            fns = os.listdir(folder)
            for fn in fns:
                completefn = os.path.join(folder, fn)
                if os.path.isfile(completefn) and not completefn.endswith(".tif"):
                    os.rename(completefn, completefn + ".tif")

    return 0

rttif("/mnt/data/feature_extraction/imgnet/")