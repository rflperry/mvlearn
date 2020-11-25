<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Contributing to multiview
=======
# Contributing to mvlearn
>>>>>>> master

(adopted from scikit-learn)

## Submitting a bug report or a feature request
<<<<<<< HEAD
=======
Contributing to multiview
======================

(adopted from scikit-learn)

Submitting a bug report or a feature request
--------------------------------------------
>>>>>>> Contributing draft
=======
# Contributing to multiview

(adopted from scikit-learn)

## Submitting a bug report or a feature request
>>>>>>> Contributing draft
=======
>>>>>>> master

We use GitHub issues to track all bugs and feature requests; feel free to open
an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a
ticket to the
<<<<<<< HEAD
<<<<<<< HEAD
`Bug Tracker <https://github.com/neurodata/multiview/issues>`_. You are
=======
`Bug Tracker <https://github.com/neurodata/mvlearn/issues>`_. You are
>>>>>>> master
=======
`Bug Tracker <https://github.com/mvlearn/mvlearn/issues>`_. You are
>>>>>>> master
also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
<<<<<<< HEAD
<<<<<<< HEAD
   `issues <https://github.com/neurodata/multiview/issues?q=>`_
   or `pull requests <https://github.com/neurodata/multiview/pulls?q=>`_.
=======
   `issues <https://github.com/neurodata/mvlearn/issues?q=>`_
   or `pull requests <https://github.com/neurodata/mvlearn/pulls?q=>`_.
>>>>>>> master
=======
   `issues <https://github.com/mvlearn/mvlearn/issues?q=>`_
   or `pull requests <https://github.com/mvlearn/mvlearn/pulls?q=>`_.
>>>>>>> master

-  If you are submitting a bug report, we strongly encourage you to follow the guidelines in
   :ref:`filing_bugs`.

.. _filing_bugs:

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
### How to make a good bug report
=======
How to make a good bug report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>>>>>> Contributing draft
=======
### How to make a good bug report
>>>>>>> Contributing draft

When you submit an issue to `Github
<https://github.com/neurodata/multiview/issues>`__, please do your best to
=======
### How to make a good bug report

When you submit an issue to `Github
<<<<<<< HEAD
<https://github.com/neurodata/mvlearn/issues>`__, please do your best to
>>>>>>> master
=======
<https://github.com/mvlearn/mvlearn/issues>`__, please do your best to
>>>>>>> master
follow these guidelines! This will make it a lot easier to provide you with good
feedback:

- The ideal bug report contains a **short reproducible code snippet**, this way
  anyone can try to reproduce the bug easily (see `this
  <https://stackoverflow.com/help/mcve>`_ for more details). If your snippet is
  longer than around 50 lines, please link to a `gist
  <https://gist.github.com>`_ or a github repo.

- If not feasible to include a reproducible snippet, please be specific about
  what **estimators and/or functions are involved and the shape of the data**.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
<<<<<<< HEAD
  your **Python and multiview versions**. This information
=======
  your **Python and mvlearn versions**. This information
>>>>>>> master
  can be found by running the following code snippet::

    import platform; print(platform.platform())
    import sys; print("Python", sys.version)
<<<<<<< HEAD
    import multiview; print("multiview", multiview.__version__)
=======
    import mvlearn; print("mvlearn", mvlearn.__version__)
>>>>>>> master

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**.  See `Creating and highlighting code blocks
  <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_
  for more details.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
## Contributing Code
=======
Contributing Code
-----------------
>>>>>>> Contributing draft
=======
## Contributing Code
>>>>>>> Contributing draft

The preferred workflow for contributing to multiview is to fork the main
repository on GitHub, clone, and develop on a branch. Steps: 

1. Fork the `project repository <https://github.com/neurodata/multiview>`__ by clicking
=======
## Contributing Code

The preferred workflow for contributing to mvlearn is to fork the main
repository on GitHub, clone, and develop on a branch. Steps: 

<<<<<<< HEAD
1. Fork the `project repository <https://github.com/neurodata/mvlearn>`__ by clicking
>>>>>>> master
=======
1. Fork the `project repository <https://github.com/mvlearn/mvlearn>`__ by clicking
>>>>>>> master
   on the ‘Fork’ button near the top right of the page. This creates a copy
   of the code under your GitHub user account. For more details on how to
   fork a repository see `this
   guide <https://help.github.com/articles/fork-a-repo/>`__.

<<<<<<< HEAD
2. Clone your fork of the multiview repo from your GitHub account to your
=======
2. Clone your fork of the mvlearn repo from your GitHub account to your
>>>>>>> master
   local disk:

   .. code:: bash

<<<<<<< HEAD
      $ git clone git@github.com:YourLogin/multiview.git
      $ cd multiview
=======
      $ git clone git@github.com:YourLogin/mvlearn.git
      $ cd mvlearn
>>>>>>> master

3. Create a ``feature`` branch to hold your development changes:

   .. code:: bash

      $ git checkout -b my-feature

   Always use a ``feature`` branch. It’s good practice to never work on
   the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using
   ``git add`` and then ``git commit`` files:

   .. code:: bash

      $ git add modified_files
      $ git commit

   to record your changes in Git, then push the changes to your GitHub
   account with:

   .. code:: bash

      $ git push -u origin my-feature

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
### Pull Request Checklist
=======
Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~
>>>>>>> Contributing draft
=======
### Pull Request Checklist
>>>>>>> Contributing draft
=======
### Pull Request Checklist
>>>>>>> master

We recommended that your contribution complies with the following rules
before you submit a pull request: 

-  Follow the `coding-guidelines <#guidelines>`__. 
-  Give your pull request a helpful title that summarises what your contribution does. 
   In some cases ``Fix <ISSUE TITLE>`` is enough. ``Fix #<ISSUE NUMBER>`` is not enough.
-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate. 
-  At least one paragraph of narrative documentation with links to references in 
   the literature (with PDF links when possible) and the example. 
-  All functions and classes must have unit tests. These should include, 
   at the very least, type checking and ensuring correct computation/outputs.
-  Ensure all tests are passing locally using ``pytest``. Install the necessary
   packages by: 

   .. code:: bash

      $ pip install pytest pytest-cov

   then run

   .. code:: bash
   
      $ pytest

   or you can run pytest on a single test file by

   .. code:: bash
   
      $ pytest path/to/test.py

-  Run an autoformatter to conform to PEP 8 style guidelines. We use ``black`` and would like for you to format all files using ``black``. You can run the following lines to format your files.

   .. code:: bash

      $ pip install black
      $ black path/to/module.py

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
## Guidelines

### Coding Guidelines
=======
Guidelines
----------

Coding Guidelines
~~~~~~~~~~~~~~~~~
>>>>>>> Contributing draft
=======
## Guidelines

### Coding Guidelines
>>>>>>> Contributing draft

Uniformly formatted code makes it easier to share code ownership. multiview
=======
## Guidelines

### Coding Guidelines

Uniformly formatted code makes it easier to share code ownership. mvlearn
>>>>>>> master
package closely follows the official Python guidelines detailed in
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ that detail how
code should be formatted and indented. Please read it and follow it.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
### Docstring Guidelines
=======
Docstring Guidelines
~~~~~~~~~~~~~~~~~~~~
>>>>>>> Contributing draft
=======
### Docstring Guidelines
>>>>>>> Contributing draft
=======
### Docstring Guidelines
>>>>>>> master

Properly formatted docstrings is required for documentation generation
by Sphinx. The pygraphstats package closely follows the numpydoc
guidelines. Please read and follow the
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`__
guidelines. Refer to the
`example.py <https://numpydoc.readthedocs.io/en/latest/example.html#example>`__
provided by numpydoc.

<<<<<<< HEAD
## API of multiview Objects

### Estimators

The main multiview object is the estimator and its documentation draws mainly from the formatting of sklearn’s estimator object. An estimator is an object that fits a set of training data and generates some new view of the data. In contributing, borrow from sklearn requirements as much as possible and utilize their checks to automatically check the suitability of inputted data.
=======
## API of mvlearn Objects

### Estimators

The main mvlearn object is the estimator and its documentation draws mainly from the formatting of sklearn’s estimator object. An estimator is an object that fits a set of training data and generates some new view of the data. In contributing, borrow from sklearn requirements as much as possible and utilize their checks to automatically check the suitability of inputted data.
>>>>>>> master

#### Instantiation

An estimator object’s `__init__` method may accept constants that determine the behavior of the object’s methods. These constants should not be the data nor should they be data-dependent as those are left to the `fit()` method. All instantiation arguments are keyworded and have default values. Thus, the object keeps these values across different method calls. Every keyword argument accepted by `__init__` should correspond to an instance attribute and there should be no input validation logic on instantiation, as that is left to `fit`. A correct implementation of `__init__` looks like

```python
def __init__(self, param1=1, param2=2):
    self.param1 = param1
    self.param2 = param2
```



#### Fitting

All estimators implement the fit method to make some estimation, either:

```python

estimator.fit(Xs, y)

```

or

```python

estimator.fit(Xs)

```

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> master
The former case corresponds to the supervised case and the latter to the unsupervised case. In unsupervised cases, y takes on a default value of `None` and is ignored. Xs corresponds to a list of data matrices and y to a list of sample labels. The samples across views in Xs and y are matched. Note that data matrices in Xs must have the same number of samples (rows) but the number of features (columns) may differ.

| **Parameters** | **Format**                                         |
| -------------- | -------------------------------------------------- |
| Xs             | list of array-likes <br>&nbsp;&nbsp;- Xs shape: (n_views,) <br>&nbsp;&nbsp;- Xs[i] shape: (n_samples, n_features_i)           |
<<<<<<< HEAD
=======
The former case corresponds to the supervised case and the latter to the unsupervised case. In unsupervised cases, y takes on a default value of `None` and is ignored. Xs corresponds to a list of data matrices and y to a list of sample labels. The samples across views in Xs and y are matched.

| **Parameters** | **Format**                                         |
| -------------- | -------------------------------------------------- |
| Xs             | array-like, shape (n_views, n_samples, n_features) |
>>>>>>> Contributing draft
=======
>>>>>>> master
| y              | array, shape (n_samples,)                          |
| kwargs         | optional data-dependent parameters.                |

The `fit` method should return the object (`self`) so that simple one line processes can be written.

All attributed calculated in the `fit` method should be saved with a trailing underscore to distinguish them from the constants passes to `__init__`. They are overwritten every time `fit` is called.

### Additional Functionality

#### Transformer

A transformer object modifies the data it is given. An estimator may also be a transformer that learns the transformation parameters. The transformer object implements the method

```python
<<<<<<< HEAD
new_data = transformer.transform(data)
=======
new_data = transformer.transform(Xs)
>>>>>>> master
```

and if the fit method must be called first,

```python
<<<<<<< HEAD
new_data = transformer.fit_transform(data)
```

<<<<<<< HEAD
It may be more efficient in some cases to compute the latter example rather than call `fit` and `transform` separately.
=======
It may be more efficient in some cases to compute the latter example rather than call `fit` and `transform` separately.
>>>>>>> Contributing draft
=======
new_data = transformer.fit_transform(Xs, y)
```

It may be more efficient in some cases to compute the latter example rather than call `fit` and `transform` separately.
>>>>>>> master
