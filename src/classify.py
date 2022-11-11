from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


def fit_lr_classifier(X, y):
  """
  Wrapper for `sklearn.linear.model.LogisticRegression`

  Parameters
  ----------
  X : np.array, shape `(n_examples, n_features)`
      The matrix of features, one example per row.

  y : list
      The list of labels for rows in `X`.

  Returns
  -------
  Trained sklearn.linear.model.LogisticRegression model

  """
  mod = LogisticRegression(solver='liblinear')
  mod.fit(X, y)
  return mod


def fit_nb_classifier(X, y):
  """
  Wrapper for `sklearn.naive_bayes.MultinomialNB`

  Parameters
  ----------
  X : np.array, shape `(n_examples, n_features)`
      The matrix of features, one example per row.

  y : list
      The list of labels for rows in `X`.

  Returns
  -------
  Trained sklearn.naive_bayes.MultinomialNB model

  """
  mod = MultinomialNB()
  mod.fit(X, y)
  return mod


def fit_svm_classifier(X, y):
  """
  Wrapper for `sklearn.svm.LinearSVC`

  Parameters
  ----------
  X : np.array, shape `(n_examples, n_features)`
      The matrix of features, one example per row.

  y : list
      The list of labels for rows in `X`.

  Returns
  -------
  Trained sklearn.svm.LinearSVC model

  """
  mod = LinearSVC()
  mod.fit(X, y)
  return mod


def experiment_train_test(X_train, y_train, X_test, y_test, classifier, verbose=True):
  """
  Experimental framework. 
  Train a `classifier` on `X_train`, `y_train`.
  Predict on `X_test` using the trained `classifier`.
  Score predictions with f1_score.
  Print a classifiection report, if verbose=True.
  Return model, f1_score, predictions

  Parameters
  ----------
  X_train : np.array, shape `(n_examples, n_features)`
      The matrix of features, one example per row.

  y_train : list
      The list of labels for rows in `X_train`.
  
  X_test : np.array, shape `(n_examples, n_features)`
      The matrix of features, one example per row.

  y_test : list
      The list of labels for rows in `X_test`.

  classifier : str ['nb', 'svm', or 'lr']
      naive Bayes, support vector machine, logistric regression

  verbose : bool (default: True)
      Whether to print out the model assessment to standard output.
      Set to False for statistical testing via repeated runs.

  Prints
  -------
  To standard output, if `verbose=True`
      classification report of precision/recall/f1.
  Returns
  -------
  dict with keys
      'model': trained model
      'predictions': predictions on the test dataset
      'score':f1 score on test dataset

  """
  # train
  if classifier == 'nb':
    mod = fit_nb_classifier(X_train, y_train)
  elif classifier == 'svm':
    mod = fit_svm_classifier(X_train, y_train)
  elif classifier == 'lr':
    mod = fit_lr_classifier(X_train, y_train)
  else:
    print(f"[{classifier}] is not implemented,\
      please choose one among [naive Bayes, support vector machine, logistric regression]\
      by providing a string argument 'nb', 'svm', or 'lr'")

  # test
  preds = mod.predict(X_test)
  score = f1_score(y_test, preds, average='macro', pos_label=None)
  if verbose:
    print(f"{mod}")
    print(classification_report(y_test, preds, digits=3))

  return {'model': mod,
          'score': score,
          'predictions': preds}
