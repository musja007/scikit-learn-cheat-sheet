# scikit-learn-cheat-sheet
A compilation of main commands for scikit-learn with examples

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
