import os
import shutil


def collect_img_subset(root_dir, out_dir, name_list_dir, intersect=100):
    fp = open(name_list_dir, "r")
    name_list = []
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, item in enumerate(fp.readlines()):
        if not i % intersect:
            name_list.append(item.split()[0])
    fp.close()

    for name in name_list:
        folder = name.split("/")[0]
        if not os.path.exists(os.path.join(out_dir, folder)):
            os.makedirs(os.path.join(out_dir, folder))
        img_name = name.split("/")[-1]
        shutil.copy2(os.path.join(root_dir, "%s.JPEG" % name), os.path.join(out_dir, folder, "%s.jpg" % img_name))

    return 0

collect_img_subset("/mnt/data/ILSVRC/Data/CLS-LOC/train", "/mnt/data/imgnet/train", "/mnt/data/ILSVRC/ImageSets/CLS-LOC/train_loc.txt")
collect_img_subset("/mnt/data/ILSVRC/Data/CLS-LOC/test", "/mnt/data/imgnet/test", "/mnt/data/ILSVRC/ImageSets/CLS-LOC/test.txt")
collect_img_subset("/mnt/data/ILSVRC/Data/CLS-LOC/val", "/mnt/data/imgnet/val", "/mnt/data/ILSVRC/ImageSets/CLS-LOC/val.txt")