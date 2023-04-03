# scikit-learn-cheat-sheet
A compilation of main commands for scikit-learn with examples. Inspired by https://inria.github.io/scikit-learn-mooc/index.html.

- [scikit-learn-cheat-sheet](#scikit-learn-cheat-sheet)
  * [1. Numerical data preprocessing](#1-numerical-data-preprocessing)
    + [StandardScaler](https://github.com/musja007/scikit-learn-cheat-sheet/blob/main/README.md#standardscaler)
    + [[MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)](#-minmaxscaler--https---scikit-learnorg-stable-modules-generated-sklearnpreprocessingminmaxscalerhtml-)
    + [[Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)](#-normalizer--https---scikit-learnorg-stable-modules-generated-sklearnpreprocessingnormalizerhtml-)
    + [[Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)](#-binarizer--https---scikit-learnorg-stable-modules-generated-sklearnpreprocessingbinarizerhtml-)
    + [[SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)](#-simpleimputer--https---scikit-learnorg-stable-modules-generated-sklearnimputesimpleimputerhtml-)
    + [[PolynomialFeatures¶](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)](#-polynomialfeatures---https---scikit-learnorg-stable-modules-generated-sklearnpreprocessingpolynomialfeatureshtml-)
  * [2. Encoding](#2-encoding)
    + [[OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)](#-ordinalencoder--https---scikit-learnorg-stable-modules-generated-sklearnpreprocessingordinalencoderhtml-)
    + [[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)](#-onehotencoder--https---scikit-learnorg-stable-modules-generated-sklearnpreprocessingonehotencoderhtml-)
  * [3. Column selection and transformation](#3-column-selection-and-transformation)
    + [[make_column_selector](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector)](#-make-column-selector--https---scikit-learnorg-stable-modules-generated-sklearncomposemake-column-selectorhtml-sklearncomposemake-column-selector-)
    + [[ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer)](#-columntransformer--https---scikit-learnorg-stable-modules-generated-sklearncomposecolumntransformerhtml-sklearncomposecolumntransformer-)
  * [4. Pipelines](#4-pipelines)
    + [[make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)](#-make-pipeline--https---scikit-learnorg-stable-modules-generated-sklearnpipelinemake-pipelinehtml-)
    + [[set_config](https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config)](#-set-config--https---scikit-learnorg-stable-modules-generated-sklearnset-confightml-sklearnset-config-)
  * [5. Model training](#5-model-training)
    + [[train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)](#-train-test-split--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectiontrain-test-splithtml-sklearnmodel-selectiontrain-test-split-)
    + [[learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)](#-learning-curve--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectionlearning-curvehtml-)
  * [6. Metrics](#6-metrics)
    + [[mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)](#-mean-squared-error--https---scikit-learnorg-stable-modules-generated-sklearnmetricsmean-squared-errorhtml-sklearnmetricsmean-squared-error-)
    + [[precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)](#-precision-score--https---scikit-learnorg-stable-modules-generated-sklearnmetricsprecision-scorehtml-)
    + [[recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)](#-recall-score--https---scikit-learnorg-stable-modules-generated-sklearnmetricsrecall-scorehtml-)
    + [[balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)](#-balanced-accuracy-score--https---scikit-learnorg-stable-modules-generated-sklearnmetricsbalanced-accuracy-scorehtml-sklearnmetricsbalanced-accuracy-score-)
    + [[confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)](#-confusion-matrix--https---scikit-learnorg-stable-modules-generated-sklearnmetricsconfusion-matrixhtml-sklearnmetricsconfusion-matrix-)
    + [[ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)](#-confusionmatrixdisplay--https---scikit-learnorg-stable-modules-generated-sklearnmetricsconfusion-matrixhtml-sklearnmetricsconfusion-matrix-)
    + [[roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)](#-roc-curve--https---scikit-learnorg-stable-modules-generated-sklearnmetricsroc-curvehtml-sklearnmetricsroc-curve-)
    + [[auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)](#-auc--https---scikit-learnorg-stable-modules-generated-sklearnmetricsauchtml-sklearnmetricsauc-)
    + [[RocCurveDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html)](#-roccurvedisplay--https---scikit-learnorg-stable-modules-generated-sklearnmetricsroccurvedisplayhtml-)
    + [[precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)](#-precision-recall-curve--https---scikit-learnorg-stable-modules-generated-sklearnmetricsprecision-recall-curvehtml-sklearnmetricsprecision-recall-curve-)
    + [[PrecisionRecallDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay)](#-precisionrecalldisplay--https---scikit-learnorg-stable-modules-generated-sklearnmetricsprecisionrecalldisplayhtml-sklearnmetricsprecisionrecalldisplay-)
  * [7. Parameter tuning](#7-parameter-tuning)
    + [[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)](#-gridsearchcv--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectiongridsearchcvhtml-)
    + [[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)](#-randomizedsearchcv--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectiongridsearchcvhtml-)
  * [8. Model selection](#8-model-selection)
    + [[cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)](#-cross-validate--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectioncross-validatehtml-)
    + [[cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)](#-cross-val-score--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectioncross-val-scorehtml-)
    + [[validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html)](#-validation-curve--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectionvalidation-curvehtml-)
    + [[KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)](#-kfold--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectionkfoldhtml-)
    + [[ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)](#-shufflesplit--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectionshufflesplithtml-)
    + [[StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)](#-stratifiedkfold--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectionstratifiedkfoldhtml-)
    + [[GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)](#-groupkfold--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectiongroupkfoldhtml-)
    + [[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)](#-timeseriessplit--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectiontimeseriessplithtml-)
    + [[LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)](#-leaveonegroupout--https---scikit-learnorg-stable-modules-generated-sklearnmodel-selectionleaveonegroupouthtml-)
  * [9. Dummy models](#9-dummy-models)
    + [[DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)](#-dummyregressor--https---scikit-learnorg-stable-modules-generated-sklearndummydummyregressorhtml-)
    + [[DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)](#-dummyclassifier--https---scikit-learnorg-stable-modules-generated-sklearndummydummyclassifierhtml-)
  * [10. Linear models](#10-linear-models)
    + [[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)](#-linearregression--https---scikit-learnorg-stable-modules-generated-sklearnlinear-modellinearregressionhtml-)
    + [[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)](#-ridge--https---scikit-learnorg-stable-modules-generated-sklearnlinear-modelridgehtml-)
    + [[RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)](#-ridgecv--https---scikit-learnorg-stable-modules-generated-sklearnlinear-modelridgecvhtml-)
    + [[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)](#-logisticregression--https---scikit-learnorg-stable-modules-generated-sklearnlinear-modellogisticregressionhtml-)
  * [11. kNN](#11-knn)
    + [[KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)](#-kneighborsregressor--https---scikit-learnorg-stable-modules-generated-sklearnneighborskneighborsregressorhtml-sklearnneighborskneighborsregressor-)
    + [[KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)](#-kneighborsclassifier--https---scikit-learnorg-stable-modules-generated-sklearnneighborskneighborsclassifierhtml-sklearnneighborskneighborsclassifier-)
  * [12. Tree models](#12-tree-models)
    + [[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)](#-decisiontreeregressor--https---scikit-learnorg-stable-modules-generated-sklearntreedecisiontreeregressorhtml-sklearntreedecisiontreeregressor-)
    + [[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)](#-decisiontreeclassifier--https---scikit-learnorg-stable-modules-generated-sklearntreedecisiontreeclassifierhtml-sklearntreedecisiontreeclassifier-)
  * [12. Ensemble models](#12-ensemble-models)
    + [[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)](#-gradientboostingregressor--https---scikit-learnorg-stable-modules-generated-sklearnensemblegradientboostingregressorhtml-)
    + [[GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)](#-gradientboostingclassifier--https---scikit-learnorg-stable-modules-generated-sklearnensemblegradientboostingclassifierhtml-)
    + [[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)](#-randomforestregressor--https---scikit-learnorg-stable-modules-generated-sklearnensemblerandomforestregressorhtml-)
    + [[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)](#-randomforestclassifier--https---scikit-learnorg-stable-modules-generated-sklearnensemblerandomforestclassifierhtml-)

## 1. Numerical data preprocessing

### [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

Standardizes data by removing the mean and scaling to unit variance.

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)
```

### [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

Transforms the data so that it values appear in the given range.

```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
scaler.transform(data)
```

### [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)

Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1, l2 or inf) equals one.

```
from sklearn.preprocessing import Normalizer
transformer = Normalizer()
transformer.fit(data)
transformer.transform(data)
```

### [Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)

Binarizes data (set feature values to 0 or 1) according to a threshold.

```
from sklearn.preprocessing import Binarizer
transformer = Binarizer().fit(data)
transformer.transform(data)
```

### [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

Replaces missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.
Parameters: `missing_values` specifies what we assume as a missing value, `strategy` specifies what we will replace the missing values with.

```
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(data)
imputer.transform(data)
```

### [PolynomialFeatures¶](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

Generates polynomial and interaction features.
Parameters: `degree` specifies the maximal degree of the polynomial features

```
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
poly.fit_transform(data)
```

## 2. Encoding

### [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)

`OrdinalEncoder` will encode each category with a different number.

```
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data)
```

### [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

For a given feature, `OneHotEncoder` will create as many new columns as there are possible categories. For a given sample, the value of the column corresponding to the category will be set to 1 while all the columns of the other categories will be set to 0.

```
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
data_encoded = encoder.fit_transform(data)
```

## 3. Column selection and transformation

### [make_column_selector](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector)

Selects columns based on datatype or the columns name.

### [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer)

Applies specific transformations to the subset of columns in the data.

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer([
    ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)
```

## 4. Pipelines

### [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)

Allows to construct a pipeline – a set of commands/models/etc. which will be executed consequently.

```
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
)
```

### [set_config](https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config)

Allows to vizualize the pipelines in Jupyter, needs to be set once at the beginning of your notebook.

```
from sklearn import set_config
set_config(display="diagram")
```

## 5. Model training

### [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)

Split arrays or matrices into random train and test subsets.

```
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)
```

### [learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)

Allows to see how the model performance changes when choosing different train/test split size.

```
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=30, test_size=0.2)

from sklearn.model_selection import learning_curve

train_sizes=[0.3, 0.6, 0.9]
results = learning_curve(
    regressor, data, target, train_sizes=train_sizes, cv=cv,
    scoring="neg_mean_absolute_error", n_jobs=2)
```

## 6. Metrics

### [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)

```
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)
```

### [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

`average` parameter is required for multiclass/multilabel targets.

```
from sklearn.metrics import precision_score
precision_score(y_true, y_pred, average='macro')
```

### [recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

`average` parameter is required for multiclass/multilabel targets.

```
from sklearn.metrics import recall_score
recall_score(y_true, y_pred, average='macro')
```

### [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)

The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.

```
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_true, y_pred)
```

### [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)

```
from sklearn.metrics import confusion_matrix
labels=["a", "b", "c"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
```

### [ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)

Confusion Matrix visualization.

```
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels=["a", "b", "c"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()
```
### [roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)

`pos_label` parameter defines the label of the positive class.

```
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
```

### [auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)

Compute Area Under the Curve (AUC).

```
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
roc_auc = auc(fpr, tpr)
```

### [RocCurveDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html)

ROC Curve visualization.

```
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
roc_auc = auc(fpr, tpr), estimator_name='example estimator')
disp.plot()
plt.show()
```

### [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)

```
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
```

### [PrecisionRecallDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay)

Precision-Recall visualization.

```
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()
```

## 7. Parameter tuning

### [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

Greedy search over specified parameter values for an estimator.

```
from sklearn.model_selection import GridSearchCV
param_grid = {
    'parameter_A': (0.01, 0.1, 1, 10),
    'parameter_B': (3, 10, 30)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=2, cv=2)
model_grid_search.fit(data, target)
```

### [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

In contrast to `GridSearchCV`, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by `n_iter`.

```
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'parameter_A': (0.01, 0.1, 1, 10),
    'parameter_B': (3, 10, 30)}
model_random_search = RandomizedSearchCV(
    model, param_distributions=param_grid, n_iter=10,
    cv=5, verbose=1,
)
model_random_search.fit(data, target)
```

## 8. Model selection

### [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)

Evaluate metric(s) by cross-validation and also record fit/score times. `scoring` parameters is used to define which metric(s) will be computed during each fold. In the `cv` parameter, one can pass any type of splitting strategy: k-fold, stratified and etc.

```
from sklearn.model_selection import cross_validate
cv_results = cross_validate(
    model, data, target, cv=5, scoring="neg_mean_absolute_error")
```

### [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

Identical to calling the `cross_validate` function and to select the test score only.

```
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data, target)
```

### [validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html)

Determine training and test scores for varying parameter values.

```
from sklearn.model_selection import validation_curve
param_A = [1, 5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    model, data, target, param_name="param_A", param_range=param_A,
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
```

### [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

K-Folds cross-validator.

```
from sklearn.model_selection import KFold
cv = KFold(n_splits=2)
cv..get_n_splits(data)
```
### [ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)

Random permutation cross-validator.

```
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, random_state=0)
cv.get_n_splits(data)
```

### [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

Stratified K-Folds cross-validator, generates test sets such that all contain the same distribution of classes, or as close as possible.

```
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=2)
cv.get_n_splits(data, target)
```

### [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)

K-fold iterator variant with non-overlapping groups, which makes each group appear exactly once in the test set across all folds. `groups` should be an array of the same length of data. For each row `groups` should indicate which group it belongs to.

```
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=2)
cv.get_n_splits(data, target, groups=groups)
```

### [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

Time Series cross-validator, provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets.

```
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=2)
cv.get_n_splits(data, target)
```

### [LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)

Leave One Group Out cross-validator, provides train/test indices to split data such that each training set is comprised of all samples except ones belonging to one specific group. `groups` should be an array of the same length of data. For each row `groups` should indicate which group it belongs to.

```
from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()
cv.get_n_splits(data, target, groups=groups)
```

## 9. Dummy models

### [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)

Predicts the same value based on a (simple) rule without using training features. `strategy` can be `{“mean”, “median”, “quantile”, “constant”}`.

```
from sklearn.dummy import DummyRegressor
model = DummyRegressor(strategy="mean")
model.fit(data, target)
```

### [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)

Predicts the same class based on a (simple) rule without using training features. `strategy` can be `{“most_frequent”, “prior”, “stratified”, “uniform”, “constant”}`.

```
from sklearn.dummy import DummyClassifier
model = DummyClassifier(strategy="most_frequent")
model.fit(data, target)
```

## 10. Linear models

### [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

Ordinary least squares Linear Regression.

```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data, target)
```

### [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

Linear least squares with l2 regularization. `alpha` parameter defines the l2 multiplier coefficient.

```
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(data, target)
```

### [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)

Ridge regression with built-in cross-validation. `alphas` defines the array of alpha values to try.

```
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
model.fit(data, target)
```

### [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

Logistic Regression classifier. `penalty` parameter is by default `l2`, `C` defines inverse of regularization strength, must be a positive float.

```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C = 1.0)
model.fit(data, target)
```

## 11. kNN

### [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)

Regression based on k-nearest neighbors. `n_neighbors` defines number of neighbors to use.

```
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=2)
model.fit(data, target)
```

### [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

Regression based on k-nearest neighbors. `n_neighbors` defines number of neighbors to use.

```
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsRegressor(n_neighbors=2)
model.fit(data, target)
```

## 12. Tree models

### [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)

A decision tree regressor. `max_depth` defines the maximum depth of a tree.

```
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=2)
model.fit(data, target)
```

### [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

A decision tree classifier. `max_depth` defines the maximum depth of a tree.

```
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)
model.fit(data, target)
```

## 12. Ensemble models

### [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

Gradient Boosting trees for regression. `max_depth` defines the maximum depth of a tree, `learning_rate` defines the "contribution" of each tree, `n_estimators` controls the number of trained trees.

```
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(max_depth=2, learning_rate=1.0, n_estimators=100)
model.fit(data, target)
```

### [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

Gradient Boosting for classification. `max_depth` defines the maximum depth of a tree, `learning_rate` defines the "contribution" of each tree, `n_estimators` controls the number of trained trees.

```
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingRegressor(max_depth=2, learning_rate=1.0, n_estimators=100)
model.fit(data, target)
```

### [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

A random forest regressor. `max_depth` defines the maximum depth of a tree, `n_estimators` controls the number of trained trees.

```
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=2, n_estimators=100)
model.fit(data, target)
```

### [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

A random forest classifier. `max_depth` defines the maximum depth of a tree, `n_estimators` controls the number of trained trees.

```
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=2, n_estimators=100)
model.fit(data, target)
```
