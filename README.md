# scikit-learn-cheat-sheet
A compilation of main commands for scikit-learn with examples

## 1. Preprocessing

### [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

Standardizes data by removing the mean and scaling to unit variance.

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)
```
