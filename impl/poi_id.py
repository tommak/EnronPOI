#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from preprocess import preprocess, ordered_columns

import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_selection import  SelectPercentile, mutual_info_classif, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def load_or_fit(clf, features, labels, path, dump_new=True, features_path=None):
    """
    Load dumped version of classifier from provided path.
    Otherwise fit classifier, if dump_new is True, then dump it in a provided path.
    If features_path is specified, than dump current features_list to the features_path
    """
    
    try:
        with open(path, "r") as clf_infile:
            fitted_clf = pickle.load(clf_infile)
        print "Classifier was loaded from ", path
    except IOError:
        print "Failed to load fitted classifier\nStart fitting..."
        t0 = time()
        fitted_clf = clf
        fitted_clf.fit(features, labels)
        print "Classifier fitted in ", round(time()-t0, 3), "s"
        if dump_new:
            with open(path, "w") as clf_outfile:
                pickle.dump(fitted_clf, clf_outfile)
            print "Fitted classifier was dumped to ", path
        
    if features_path:
        with open(features_path, "w") as feat_outfile:
            pickle.dump(features_list, feat_outfile)
    
    return fitted_clf 

if __name__ == "__main__":

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)


	### Clean data and add new features
	my_dataset = preprocess(data_dict)

	### Define what features to use.
	exclude = ["poi", "email_address"]
	features_list = ["poi"] + [f for f in my_dataset.items()[0][1].keys() if f not in exclude]

	### Format data for further analysis
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	### Define classification pipeline 
	base_clf = Pipeline([
                    ('scaling', MinMaxScaler()),     
                    ('feature_selection', SelectPercentile()),
                    ('classification', LogisticRegression(class_weight="balanced"))
                  ])

	### Define parameters for tuning and run grid search
	param = {
            "feature_selection__score_func" : [f_classif, mutual_info_classif],
            "feature_selection__percentile" : [30, 50, 70, 100],
            "classification__C" : np.logspace(-2,2,num=5,endpoint=True)
            }


	clf_search = GridSearchCV(base_clf,
                    param_grid = param,
                    scoring = make_scorer(f1_score),
                    cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=32))    

	clf_search = load_or_fit(clf_search, features, labels, "my_classifier_grsearch.pkl")
	
	### Use algorithm with highest score as a final classifier
	clf = clf_search.best_estimator_

	### Dump final classifier, dataset, and features_list 
	dump_classifier_and_data(clf, my_dataset, features_list)
