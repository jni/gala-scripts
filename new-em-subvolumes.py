# IPython log file


from gala import imio
import numpy as np

slices = [(slice(None), slice(None, 625), slice(None, 625)),
          (slice(None), slice(None, 625), slice(625, None)),
          (slice(None), slice(625, None), slice(None, 625)),
          (slice(None), slice(625, None), slice(625, None))]

gt = imio.read_h5_stack('ground-truth.h5', group='bodies')
gts = [gt[s] for s in slices]
from skimage.measure import label
for i, vol in enumerate(gts):
    fn = 'ground-truth-%i.lzf.h5' % i
    vol_relabel = label(vol)
    print(np.max(vol_relabel))
    imio.write_h5_stack(vol_relabel.astype(np.uint16), fn,
                        compression='lzf')

pr = imio.read_image_stack('membrane/*.tiff')
prs = [pr[s] for s in slices]
for i, vol in enumerate(prs):
    fn = 'probabilities-%i.lzf.h5' % i
    imio.write_h5_stack(vol.astype(np.uint8), fn, compression='lzf')
