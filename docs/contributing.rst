Contributing to mvlearn
=========================

(adopted from scikit-learn)

Submitting a bug report or a feature request
--------------------------------------------

We use GitHub issues to track all bugs and feature requests; feel free
to open an issue if you have found a bug or wish to see a feature
implemented.

In case you experience issues using this package, do not hesitate to
submit a ticket to the
`Bug Tracker <https://github.com/mvlearn/mvlearn/issues>`_. You
are also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the following
rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/mvlearn/mvlearn/issues?q=>`_ or
   `pull requests <https://github.com/mvlearn/mvlearn/pulls?q=>`_.

-  If you are submitting a bug report, we strongly encourage you to
   follow the guidelines in :ref:`filing_bugs`.

-  Always make sure your code follows the general :ref:`guidelines` 
   and adheres to the :ref:`api_of_mvlearn_objects`.

.. _filing_bugs:

How to make a good bug report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you submit an issue to
`Github <https://github.com/mvlearn/mvlearn/issues>`_, please
do your best to follow these guidelines! This will make it a lot easier
to provide you with good feedback:

-  The ideal bug report contains a **short reproducible code snippet**,
   this way anyone can try to reproduce the bug easily (see
   `this <https://stackoverflow.com/help/mcve>`_ for more details).
   If your snippet is longer than around 50 lines, please link to a
   `gist <https://gist.github.com>`_ or a github repo.

-  If not feasible to include a reproducible snippet, please be specific
   about what **estimators and/or functions are involved and the shape
   of the data**.

-  If an exception is raised, please **provide the full traceback**.

-  Please include your **operating system type and version number**, as
   well as your **Python and mvlearn versions**. This information can
   be found by running the following code snippet in Python.

   .. code-block:: python

      import platform; print(platform.platform());
      import sys; print("Python", sys.version);
      import mvlearn; print("mvlearn", mvlearn.version)


-  Please ensure all **code snippets and error messages are formatted in
   appropriate code blocks**. See
   `Creating and highlighting code blocks <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_
   for more details.

Contributing Code
-----------------

The preferred workflow for contributing to mvlearn is to fork the main
repository on GitHub, clone, and develop on a branch. Steps:

1. Fork the
   `project repository <https://github.com/mvlearn/mvlearn>`_
   by clicking on the ‘Fork’ button near the top right of the page. This
   creates a copy of the code under your GitHub user account. For more
   details on how to fork a repository see
   `this guide <https://help.github.com/articles/fork-a-repo/>`_.


2. Clone your fork of the mvlearn repo from your GitHub account to
   your local disk:

   .. code:: bash

       $ git clone git@github.com:YourLogin/mvlearn.git
       $ cd mvlearn


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

Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~

We recommended that your contribution complies with the following rules
before you submit a pull request:

-  Follow the `coding-guidelines <#guidelines>`__.

-  Give your pull request a helpful title that summarises what your
   contribution does. In some cases ``Fix <ISSUE TITLE>`` is enough.
   ``Fix #<ISSUE NUMBER>`` is not enough.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and the
   example.

-  All functions and classes must have unit tests. These should include,
   at the very least, type checking and ensuring correct
   computation/outputs.

-  Ensure all tests are passing locally using ``pytest``. Install the
   necessary packages by:

   .. code:: bash

       $ pip install pytest pytest-cov

   then run

   .. code:: bash

       $ pytest

   or you can run pytest on a single test file by

   .. code:: bash

       $ pytest path/to/test.py

-  Run an autoformatter to conform to PEP 8 style guidelines. We use
   ``black`` and would like for you to format all files using ``black``.
   You can run the following lines to format your files.

   .. code:: bash

       $ pip install black
       $ black path/to/module.py

.. _guidelines:

Guidelines
----------

Coding Guidelines
~~~~~~~~~~~~~~~~~

Uniformly formatted code makes it easier to share code ownership.
mvlearn package closely follows the official Python guidelines
detailed in `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__
that detail how code should be formatted and indented. Please read it
and follow it.

Docstring Guidelines
~~~~~~~~~~~~~~~~~~~~

Properly formatted docstrings is required for documentation generation
by Sphinx. The pygraphstats package closely follows the numpydoc
guidelines. Please read and follow the
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`__
guidelines. Refer to the
`example.py <https://numpydoc.readthedocs.io/en/latest/example.html#example>`__
provided by numpydoc.

.. _api_of_mvlearn_objects:

API of mvlearn Objects
----------------------

Estimators
~~~~~~~~~~

The main mvlearn object is the estimator and its documentation draws
mainly from the formatting of sklearn’s estimator object. An estimator
is an object that fits a set of training data and generates some new
view of the data. Each module in mvlearn contains a main base class
(found in ``module_name.base``) which all estimators in that module
should implement. Each of these base classes implements
sklearn.base.BaseEstimator. If you are contributing a new estimator,
be sure that it properly implements the base class
of the module it is contained within.

When contributing, borrow from sklearn requirements as
much as possible and utilize their checks to automatically check the
suitability of inputted data, or use the checks available in
``mvlearn.utils`` such as ``check_Xs``.

Instantiation
^^^^^^^^^^^^^

An estimator object’s ``__init__`` method may accept constants that
determine the behavior of the object’s methods. These constants should
not be the data nor should they be data-dependent as those are left to
the ``fit`` method. All instantiation arguments are keyworded and have
default values. Thus, the object keeps these values across different
method calls. Every keyword argument accepted by ``__init__`` should
correspond to an instance attribute and there should be no input
validation logic on instantiation, as that is left to ``fit``. A correct
implementation of ``__init__`` looks like

.. code:: python

   def __init__(self, param1=1, param2=2):
       self.param1 = param1
       self.param2 = param2

Fitting
^^^^^^^

All estimators should implement the ``fit(Xs, y=None)`` method to
make some estimation, which is called with:

.. code:: python

       estimator.fit(Xs, y)

or

.. code:: python


       estimator.fit(Xs)

The former case corresponds to the supervised case and the latter to the
unsupervised case. In unsupervised cases, y takes on a default value of
``None`` and is ignored. Xs corresponds to a list of data matrices and y
to a list of sample labels. The samples across views in Xs and y are
matched. Note that data matrices in Xs must have the same number of
samples (rows) but the number of features (columns) may differ.

+----------------+----------------------------------------------------+
| **Parameters** | **Format**                                         |
+================+====================================================+
| Xs             | list of array-likes:                               | 
|                |  - Xs shape: (n_views,)                            |
|                |  - Xs[i] shape: (n_samples, n_features_i)          |
+----------------+----------------------------------------------------+
| y              | array, shape (n_samples,)                          |
+----------------+----------------------------------------------------+
| kwargs         | optional data-dependent parameters.                |
+----------------+----------------------------------------------------+

The ``fit`` method should return the object (``self``) so that simple
one line processes can be written.

All attributes calculated in the ``fit`` method should be saved with a
trailing underscore to distinguish them from the constants passes to
``__init__``. They are overwritten every time ``fit`` is called.

Additional Functionality
~~~~~~~~~~~~~~~~~~~~~~~~

Transformers
^^^^^^^^^^^^

A ``transformer`` object modifies the data it is given and by default outputs
multiview data, a transformation of each input view. The object also has a
``multiview_output`` parameter in the constructor that may be set to ``False``
so as to return a single view, joint transformation instead. An estimator may
also be a transformer that learns the transformation parameters. The transformer
object implements the ``transform`` method, i.e.

.. code:: python

   Xs_transformed = transformer.transform(Xs)

which follows a call to ``fit``. One may alternatively perform both in the
single call

.. code:: python

   Xs_transformed = transformer.fit_transform(Xs, y)

It may be more efficient in some cases to compute the latter example
rather than call ``fit`` and ``transform`` separately.

Predictors
^^^^^^^^^^

Similarly, a ``predictor`` object makes predictions based on the
data it is given and outputs a single array of sample-specific predictions
based on all views. An estimator may also be a predictor that learns
the prediction parameters. The predictor object implements
the ``predict`` method, i.e.

.. code:: python

   y_predicted = predictor.predict(Xs)

which follows a call to ``fit``. One may alternatively perform both in the
single call

.. code:: python

   y_predicted = predictor.fit_predict(Xs, y)

It may be more efficient in some cases to compute the latter example
rather than call ``fit`` and ``predict`` separately.

Mergers
^^^^^^^

In some cases, it is helpful to extract a single view representation of
multiple views, as when one is integrating mvlearn estimators with an
sklearn pipeline. A ``merger`` object implements a ``transform`` method
which takes in multiview data and returns a single view representation, i.e.

.. code:: python

   X_merged = merger.transform(Xs)

which follows a call to ``fit`` and as above, one can perform both functions
in the single, potentially more efficient call

.. code:: python

   X_merged = merger.fit_transform(Xs)
