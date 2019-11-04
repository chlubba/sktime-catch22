import numpy as np
# import time
# import sys
# from sklearn.preprocessing import LabelEncoder
from catch22Forest.ensemble import catch22ForestClassifier

from sktime.datasets import load_gunpoint, load_basic_motions

def test_basic_univariate(network=catch22ForestClassifier()):
    '''
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score
    '''

    print("Start test_basic()")

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)

    hist = network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("End test_basic()")

def test_basic_multivariate(network=catch22ForestClassifier()):
    '''
    just a super basic test with basicmotions,
        load data,
        construct classifier,
        fit,
        score
    '''

    print("Start test_multivariate()")

    X_train, y_train = load_basic_motions(split='TRAIN', return_X_y=True)
    X_test, y_test = load_basic_motions(split='TEST', return_X_y=True)

    hist = network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("End test_multivariate()")

def test_pipeline(network=catch22ForestClassifier()):
    '''
    slightly more generalised test with sktime pipelines
        load data,
        construct pipeline with classifier,
        fit,
        score
    '''

    print("Start test_pipeline()")

    from sktime.pipeline import Pipeline

    # just a simple (useless) pipeline

    steps = [
        ('clf', network)
    ]
    clf = Pipeline(steps)

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)

    hist = clf.fit(X_train[:10], y_train[:10])

    print(clf.score(X_test[:10], y_test[:10]))
    print("End test_pipeline()")


def test_highLevelsktime(network=catch22ForestClassifier()):
    '''
    truly generalised test with sktime tasks/strategies
        load data, build task
        construct classifier, build strategy
        fit,
        score
    '''

    print("start test_highLevelsktime()")

    from sktime.highlevel.tasks import TSCTask
    from sktime.highlevel.strategies import TSCStrategy
    from sklearn.metrics import accuracy_score

    train = load_gunpoint(split='TRAIN')
    test = load_gunpoint(split='TEST')
    task = TSCTask(target='class_val', metadata=train)

    strategy = TSCStrategy(network)
    strategy.fit(task, train.iloc[:10])

    y_pred = strategy.predict(test.iloc[:10])
    y_test = test.iloc[:10][task.target].values.astype(np.float)
    print(accuracy_score(y_test, y_pred))

    print("End test_highLevelsktime()")

def test_network(network=catch22ForestClassifier()):
    # sklearn compatibility

    test_basic_univariate(network)
    # test_basic_multivariate(network)
    test_pipeline(network)
    test_highLevelsktime(network)


def all_networks_all_tests():

    networks = [
        catch22ForestClassifier(),
    ]

    for network in networks:
        print('\n\t\t' + network.__class__.__name__ + ' testing started')
        test_network(network)
        print('\t\t' + network.__class__.__name__ + ' testing finished')


def comparisonExperiments():

    data_dir = '/home/carl/Downloads/Univariate2018_ts/Univariate_ts/' # sys.argv[1]
    res_dir = '/home/carl/temp/' # sys.argv[2]

    complete_classifiers = [
        "catch22ForestClassifier",
    ]

    small_datasets = [
        "Beef",
        "Car",
        "Coffee",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "Fish",
        "GunPoint",
        "ItalyPowerDemand",
        "MoteStrain",
        "OliveOil",
        "Plane",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "SyntheticControl",
        "Trace",
        "TwoLeadECG",
    ]
    small_datasets = [
        "Beef",
        "Coffee",
        ]

    num_folds = 2

    import sktime.contrib.experiments as exp

    for f in range(num_folds):
        for d in small_datasets:
            for c in complete_classifiers:
                print(c, d, f)
                # try:
                exp.run_experiment(data_dir, res_dir, c, d, f)
                # except:
                #     print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    all_networks_all_tests()
    # comparisonExperiments() # TODO:nly works if experiments.py of sktime is changed to include catch22 as a classifier