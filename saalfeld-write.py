# IPython log file


import numpy as np
trees = np.zeros((4, 4), dtype=object)
import itertools
get_ipython().magic('pinfo itertools.combinations')
list(itertools.combinations(range(4), 2))
list(itertools.combinations_with_replacement(range(4), 2))
list(it.product(range(4), range(4))
)
list(itertools.product(range(4), range(4)))
import pickle
for tr, ts in itertools.product(range(4), range(4)):
    if tr != ts:
        with open('results-%i-%i.pickle' % (ts, tr), 'rb') as fin:
            trees[tr, ts] = pickle.load(fin)
            
t = trees[0, 1]
m01 = t.get_map(0, 1)
m01 = t.get_map(0.5)
len(np.unique(m01))
len(m01)
from gala import imio
wss = list(map(imio.read_image_stack, ['watershed-%i.lzf.h5' % i for i in range(4)]))
images = imio.read_image_stack('/groups/saalfeld/saalfeldlab/concha/sample_A/crop/raw/*.tif')
images = imio.read_image_stack('/groups/saalfeld/saalfeldlab/concha/sample_A/crop/raw/*.tiff')
images.shape
wss[0].shape
maps = [t.get_map(0.5) for t in [trees[3, 0], trees[2, 1], trees[1, 2], trees[0, 3]]]
segs = [m[ws] for ws in wss]
segs = [m[ws] for m, ws in zip(maps, wss)]
len(maps[0])
np.max(wss[0])
list(map(len, maps))
list(map(np.max, wss))
trees = trees.T
maps = maps[::-1]
segs = [m[ws] for m, ws in zip(maps, wss)]
segs.dtype
segs[0].dtype
images.dtype
seg = np.zeros(images.shape, dtype=np.uint64)
seg[:, :625, :625] = segs[0]
seg[:, :625, 625:] = segs[1]
seg[:, 625:, :625] = segs[2]
seg[:, 625:, 625:] = segs[3]
np.max(segs[0])
np.max(segs[1])
seg[:, :625, 625:] = segs[1] + np.max(segs[0])
seg[:, 625:, :625] = segs[2] + np.max(segs[0]) + np.max(segs[1])
seg[:, 625:, 625:] = segs[3] + np.max(segs[0]) + np.max(segs[1]) + np.max(segs[2])
from gala import imio
imio.write_h5_stack(images, 'gala-corners-seg-50.h5', group='raw')
imio.write_h5_stack(seg, 'gala-corners-seg-50.h5', group='labels')
import h5py
f = h5py.File('gala-corners-seg-50.h5', 'a')
f['/raw'].attrs
f['/raw'].attrs['resolution'] = np.array([12., 1, 1])
f['/labels'].attrs['resolution'] = np.array([12., 1, 1])
f.close()
from gala import evaluate as ev
gts = list(map(imio.read_image_stack, ['ground-truth-%i.lzf.h5' % i for i in range(4)]))
[ev.split_vi(s, gt) for s, gt in zip(segs, gts)]
[ev.split_vi(s, gt) for s, gt in zip(wss, gts)]
def montage_labels_4x(vols):
    y, x = vols[0].shape[1:]
    newvol = np.empty((vols[0].shape[0], y, x), dtype=np.uint64)
    newvol[:, :y, :x] = vols[0]
    newvol[:, :y, x:] = vols[1] + sum(map(np.max, vols[:1]))
    newvol[:, y:, :x] = vols[2] + sum(map(np.max, vols[:2]))
    newvol[:, y:, x:] = vols[3] + sum(map(np.max, vols[:3]))
    return newvol

wsvol = montage_labels_4x(wss)
def montage_labels_4x(vols):
    y, x = vols[0].shape[1:]
    newvol = np.empty((vols[0].shape[0], 2 * y, 2 * x), dtype=np.uint64)
    newvol[:, :y, :x] = vols[0]
    newvol[:, :y, x:] = vols[1] + sum(map(np.max, vols[:1]))
    newvol[:, y:, :x] = vols[2] + sum(map(np.max, vols[:2]))
    newvol[:, y:, x:] = vols[3] + sum(map(np.max, vols[:3]))
    return newvol

wsvol = montage_labels_4x(wss)
def write_saalfeld(fn, raw, labels, res=np.array([12., 1, 1])):
    imio.write_h5_stack(raw, fn, group='raw')
    imio.write_h5_stack(labels, fn, group='labels')
    f = h5py.File(fn, 'a')
    f['/raw'].attrs['resolution'] = res
    f['/labels'].attrs['resolution'] = res
    f.close()
    
write_saalfeld('/groups/saalfeld/saalfeldlab/concha/sample_A/juan/corners-fragments.h5', images, wsvol)
[ev.split_vi(ws, s) for ws, s in zip(wss, segs)]
from gala import agglo2
get_ipython().set_next_input('bpss = [agglo2.best_segmentation');get_ipython().magic('pinfo agglo2.best_segmentation')
get_ipython().set_next_input('bpss = [agglo2.best_segmentation');get_ipython().magic('pinfo agglo2.best_segmentation')
bpss = [agglo2.best_segmentation(ws, gt) for ws, gt in zip(wss, gts)]
[ev.split_vi(s, bp) for s, bp in zip(segs, bpss)]
