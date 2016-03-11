import os
import pickle
import numpy as np
import h5py
from sklearn.externals import joblib
from gala import imio, agglo, features, classify


fman = features.default.snemi3d()

def train(index):
    outfn = 'training-data-%i.h5' % index
    if os.path.exists(outfn):
        data, labels = classify.load_training_data_from_disk(outfn,
                                                             names=['data',
                                                                    'labels'])
    else:
        ws_tr = imio.read_image_stack('watershed-%i.lzf.h5' % index)
        pr_tr = imio.read_image_stack('probabilities-inv-norm-%i.lzf.h5' % index)
        gt_tr = imio.read_image_stack('ground-truth-%i.lzf.h5' % index)
        g = agglo.Rag(ws_tr, pr_tr,
                      feature_manager=fman)
        data, labels = g.learn_agglomerate(gt_tr, fman, min_num_epochs=4)[0][:2]
        classify.save_training_data_to_disk([data, labels],
                                            fn=outfn,
                                            names=['data', 'labels'])
    print('total training data:', data.shape)
    print('size in MB:', data.size * data.itemsize / 1e6)
    rf = classify.DefaultRandomForest(n_jobs=6)
    rf.fit(data, labels[:, 0])
    policy = agglo.classifier_probability(fman, rf)
    return policy


def test(index, policy):
    ws = imio.read_image_stack('watershed-%i.lzf.h5' % index)
    pr = imio.read_image_stack('probabilities-inv-norm-%i.lzf.h5' % index)
    g = agglo.Rag(ws, pr, merge_priority_function=policy,
                  feature_manager=fman)
    g.agglomerate(np.inf)
    return g.tree


def train_test_pair(training_index, testing_index):
    print('training %i' % training_index)
    policy = train(training_index)
    print('testing %i' % testing_index)
    tree = test(testing_index, policy)
    with open('results-%i-tr%i.pickle' % (testing_index, training_index),
              'wb') as fout:
        pickle.dump(tree, fout, protocol=-1)
    return tree


def write_saalfeld(fn, raw, labels, res=np.array([12., 1, 1])):
    imio.write_h5_stack(raw, fn, group='raw')
    imio.write_h5_stack(labels, fn, group='labels')
    f = h5py.File(fn, 'a')
    f['/raw'].attrs['resolution'] = res
    f['/labels'].attrs['resolution'] = res
    f.close()


if __name__ == '__main__':
    index_pairs = [(3 - ts, ts) for ts in range(4)]
    trees = joblib.Parallel(n_jobs=4)(joblib.delayed(train_test_pair)(*p)
                                      for p in index_pairs)
    trees = dict(zip(index_pairs, trees))
    with open('results.pickle', 'wb') as fout:
        pickle.dump(trees, fout, protocol=-1)
    images = imio.read_image_stack('/groups/saalfeld/saalfeldlab/concha/sample_A/crop/raw/*.tiff')
    wss = [imio.read_image_stack('watershed-%i.lzf.h5' % i) for i in range(4)]
    maps = [t.get_map(0.5) for t in trees]
    segs = [m[ws] for m, ws in zip(maps, wss)]

    seg = np.zeros(images.shape, dtype=np.uint64)
    seg[:, :625, :625] = segs[0]
    seg[:, :625, 625:] = segs[1] + np.max(segs[0])
    seg[:, 625:, :625] = segs[2] + np.max(segs[0]) + np.max(segs[1])
    seg[:, 625:, 625:] = segs[3] + np.max(segs[0]) + np.max(segs[1]) + np.max(segs[2])

    write_saalfeld('/groups/saalfeld/saalfeldlab/concha/sample_A/juan/corners-segments2.h5',
                   images, seg)
