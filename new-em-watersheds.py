from gala import morpho
from gala import imio
import numpy as np
pr = imio.read_image_stack('membrane/*.tiff')
ws = morpho.watershed_sequence(pr / pr.max(), axis=0, connectivity=2, smooth_thresh=0.02, minimum_seed_size=2)
imio.write_h5_stack(ws, 'watershed.lzf.h5', compression='lzf')
slices = [(slice(None), slice(None, 625), slice(None, 625)),
          (slice(None), slice(None, 625), slice(625, None)),
          (slice(None), slice(625, None), slice(None, 625)),
          (slice(None), slice(625, None), slice(625, None))]
wss = [ws[s] for s in slices]
from skimage.measure import label
for i, vol in enumerate(wss):
    fn = 'watershed-%i.lzf.h5' % i
    vol_relabel = label(vol)
    print(np.max(vol_relabel))
    imio.write_h5_stack(vol_relabel, fn, compression='lzf')
    
