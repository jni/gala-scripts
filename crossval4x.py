from gala import imio, agglo, features, classify


fman = features.default.snemi3d()

def train(index):
    ws_tr = imio.read_image_stack('watershed-%i.lzf.h5' % training)
    pr_tr = imio.read_image_stack('probabilities-%i.lzf.h5' % training)
    gt_tr = imio.read_image_stack('ground-truth-%i.lzf.h5' % training)
    g = agglo.Rag(ws_tr, pr_tr,
                  feature_manager=fman)
    data, labels = g.learn_agglomerate(gt_tr, fman, min_num_epochs=4)
    rf = classify.DefaultRandomForest()
    rf.fit(data, labels[:, 0])
    policy = agglo.classifier_priority(fmap, rf)
    return policy


def test(index, policy):
    pass


if __name__ == '__main__':
    vis = {}
    for training_index in range(4):
        policy = train(training_index)
        for testing_index in range(4):
            if testing_index == training_index:
                continue
            vi = test(testing_index, policy)
            vis[(training_index, testing_index)] = vi
    import pickle
    with open('results', 'wb') as fout:
        pickle.dump(vis, fout, protocol=-1)