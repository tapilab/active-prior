### Experimental Log ###

#### Importing data ###

`make_data.py` contains routines for importing various data sources. E.g., to create and pickle 20 newsgroup data:

	python -m active-prior.make_data --name news --output /data/active-prior/news.pkl
	2013-11-06 10:38:11-root-INFO: Namespace(name='news', output='/data/active-prior/news.pkl')
	2013-11-06 10:38:15-root-INFO: saving news data with 1177 training and 784 testing instances and 2 labels

Similarly for digits data:

	python -m active-prior.make_data --name digits --output /data/active-prior/digits.pkl
	2013-11-06 10:38:55-root-INFO: Namespace(name='digits', output='/data/active-prior/digits.pkl')
	2013-11-06 10:38:55-root-INFO: saving digits data with 200 training and 160 testing instances and 2 labels

#### Baselines ####

We can run the default experiment  to compare various active learning algorithms on the 20 newsgroup data:

	python -m active-prior.expt

This creates a plot like this:

![baseline](baseline.png)
