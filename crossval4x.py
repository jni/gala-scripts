import numpy as np
from gala import imio, agglo, features, classify


fman = features.default.snemi3d()

def train(index):
    ws_tr = imio.read_image_stack('watershed-%i.lzf.h5' % index)
    pr_tr = imio.read_image_stack('probabilities-%i.lzf.h5' % index) / 255
    gt_tr = imio.read_image_stack('ground-truth-%i.lzf.h5' % index)
    g = agglo.Rag(ws_tr, pr_tr,
                  feature_manager=fman)
    data, labels = g.learn_agglomerate(gt_tr, fman, min_num_epochs=4)[0][:2]
    rf = classify.DefaultRandomForest()
    rf.fit(data, labels[:, 0])
    policy = agglo.classifier_priority(fman, rf)
    return policy


def test(index, policy):
    ws = imio.read_image_stack('watershed-%i.lzf.h5' % index)
    pr = imio.read_image_stack('probabilities-%i.lzf.h5' % index) / 255
    g = agglo.Rag(ws, pr, merge_priority_function=policy,
                  feature_manager=fman)
    g.agglomerate(np.inf)
    return g.tree


if __name__ == '__main__':
    trees = {}
    for training_index in range(4):
        print('training %i' % training_index)
        policy = train(training_index)
        for testing_index in range(4):
            if testing_index == training_index:
                continue
            print('testing %i' % testing_index)
            tree = test(testing_index, policy)
            trees[(training_index, testing_index)] = tree
    import pickle
    with open('results', 'wb') as fout:
        pickle.dump(trees, fout, protocol=-1)
