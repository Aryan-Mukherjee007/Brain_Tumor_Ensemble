#Imports
!pip install lifelines
!pip install scikit-survival openpyxl --quiet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.inspection import permutation_importance
from sksurv.metrics import concordance_index_censored

#Running the RSF model

#Loading the dataset
try:
    file_path = 'pone.0148733.s001.xlsx'
    df = pd.read_excel(file_path, sheet_name='data')
    df.columns = df.columns.str.lower()
#Below is added as a reminder to change the above name if the file is renamed (for readers who run this)
except FileNotFoundError:
    exit()

#Dropping the missing row (hence 88 --> 87 patients)
if 'diagnosis' in df.columns and df['diagnosis'].isnull().any():
    df.dropna(subset=['diagnosis'], inplace=True)

#Features exclude whether it was censored and survival time (and id, for our purposes)
drop_cols = ['id', 'status', 'os']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[['status', 'os']]

#Defining the logrank test function for splitting nodes
def logrank_test(times_a, times_b, events_a, events_b):
    times = np.concatenate([times_a, times_b])
    events = np.concatenate([events_a, events_b])
    groups = np.concatenate([np.zeros(len(times_a)), np.ones(len(times_b))])
    sorted_indices = np.argsort(times)
    times, events, groups = times[sorted_indices], events[sorted_indices], groups[sorted_indices]

    unique_times = np.unique(times[events == 1])
    o_e_a = 0
    var_a = 0

    for t in unique_times:
        at_risk = times >= t
        n_at_risk = at_risk.sum()
        events_at_t = (times == t) & (events == 1)
        d_at_t = events_at_t.sum()
        n_a_at_risk = (groups == 0)[at_risk].sum()
        o_a_at_t = (events_at_t & (groups == 0)).sum()
        e_a_at_t = d_at_t * (n_a_at_risk / n_at_risk) if n_at_risk > 0 else 0
        o_e_a += (o_a_at_t - e_a_at_t)

        if n_at_risk > 1:
            v = (d_at_t * (n_a_at_risk / n_at_risk) * (1 - n_a_at_risk / n_at_risk) * (n_at_risk - d_at_t) / (n_at_risk - 1))
            var_a += v

    return (o_e_a ** 2) / var_a if var_a > 0 else 0

#Creating a decision tree class to containg function finding best split at each tree
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(X) <= self.min_samples_leaf:
            return {'leaf_value': y['os'].mean()}

        best_split = self._find_best_split(X, y)
        if not best_split:
            return {'leaf_value': y['os'].mean()}

        left_indices = X[best_split['feature']] <= best_split['threshold']
        right_indices = X[best_split['feature']] > best_split['threshold']

        left_subtree = self._grow_tree(X.loc[left_indices], y.loc[left_indices], depth + 1)
        right_subtree = self._grow_tree(X.loc[right_indices], y.loc[right_indices], depth + 1)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X, y):
        best_logrank_stat = -1
        best_split = None
        n_features = X.shape[1]
        features_to_test = np.random.choice(X.columns, size=int(np.sqrt(n_features)), replace=False)

        for feature in features_to_test:
            thresholds = X[feature].unique()
            if len(thresholds) > 10:
                thresholds = np.random.choice(thresholds, 10, replace=False)

            for threshold in thresholds:
                left_indices = X[feature] <= threshold
                right_indices = X[feature] > threshold
                if len(y.loc[left_indices]) < self.min_samples_leaf or len(y.loc[right_indices]) < self.min_samples_leaf:
                    continue

                test_statistic = logrank_test(
                    y.loc[left_indices, 'os'], y.loc[right_indices, 'os'],
                    y.loc[left_indices, 'status'], y.loc[right_indices, 'status']
                )

                if test_statistic > best_logrank_stat:
                    best_logrank_stat = test_statistic
                    best_split = {'feature': feature, 'threshold': threshold}
        return best_split

    def predict(self, X):
        return X.apply(self._traverse_tree, axis=1)

    def _traverse_tree(self, x, node=None):
        if node is None:
            node = self.tree
        if 'leaf_value' in node:
            return node['leaf_value']
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

#Defining a class for the out-of-bag RSF
class RandomSurvivalForest:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_leaf=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        n_samples = len(X)
        self.oob_predictions_ = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)
        original_indices = np.arange(n_samples)

        for i in range(self.n_estimators):
            print(f"Building tree {i+1}/{self.n_estimators}...", end='\r')
            bootstrap_indices = np.random.choice(original_indices, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(original_indices, np.unique(bootstrap_indices))

            X_bootstrap, y_bootstrap = X.iloc[bootstrap_indices], y.iloc[bootstrap_indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

            if len(oob_indices) > 0:
                X_oob = X.iloc[oob_indices]
                oob_preds = tree.predict(X_oob)
                self.oob_predictions_[oob_indices] += oob_preds
                oob_counts[oob_indices] += 1
        #In case of division by 0, we avoid it
        oob_counts[oob_counts == 0] = 1
        self.oob_predictions_ /= oob_counts

#C-index function
def harrells_c_index(y_true, y_pred):
    n = len(y_true)
    concordant = 0
    permissible_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            status_i, os_i = y_true['status'].iloc[i], y_true['os'].iloc[i]
            status_j, os_j = y_true['status'].iloc[j], y_true['os'].iloc[j]

            if (status_i == 1 and os_i < os_j) or (status_j == 1 and os_j < os_i):
                permissible_pairs += 1
                if (os_i < os_j and y_pred[i] < y_pred[j]) or (os_j < os_i and y_pred[j] < y_pred[i]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5

    return concordant / permissible_pairs if permissible_pairs > 0 else 0.5

#Model evaluation
rsf = RandomSurvivalForest(n_estimators=100, max_depth=5, min_samples_leaf=5)
rsf.fit(X, y)

c_index = harrells_c_index(y, rsf.oob_predictions_)
print(f"C-index for RSF: {c_index:.4f}")


#Gradient boosting

#Loading the data
data = pd.read_excel('pone.0148733.s001.xlsx', sheet_name='data')

covariates = ['Sex', 'Diagnosis', 'Location', 'KI', 'GTV', 'Stereotactic methods']
data_clean = data.dropna(subset=covariates + ['OS', 'status']).copy()

X = data_clean[covariates].astype(float)
y = data_clean['OS'].astype(float)
event = data_clean['status'].astype(bool)

#Splitting the data into 80% training, 20% testing
X_train, X_test, y_train, y_test, event_train, event_test = train_test_split(
    X, y, event, test_size=0.2, random_state=42
)

#Fitting the XGBoost Regressor
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#C-index calculation
c_idx = concordance_index(
    event_times=y_test.values,
    predicted_scores=y_pred,
    event_observed=event_test.values
)

results_data = {
    'Metric': ['Harrell\'s C-index', 'Number of Trees', 'Splits per Tree', 'Learning Rate'],
    'Value': [c_idx, 100, 8, 0.05]
}
results_table = pd.DataFrame(results_data)

#Formatting the table (for paper)
styled_table = (
    results_table
    .style
    .set_caption("Table: Gradient Boosted Regression Model Performance")
    .hide(axis='index')
    .format({'Value': '{:.4f}'})
    .set_properties(**{
        'text-align': 'center',
        'border': '1px solid black',
        'padding': '8px'
    })
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('border', '1px solid black'),
            ('background-color', '#f2f2f2'),
            ('padding', '8px')
        ]},
        {'selector': 'caption', 'props': [
            ('font-size', '13pt'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('caption-side', 'top'),
            ('margin-bottom', '10px')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
            ('border', '1px solid black')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin-left', 'auto'),
            ('margin-right', 'auto'),
            ('width', 'auto')
        ]}
    ])
)

print(f"Model trained with {len(X_train)} samples, tested on {len(X_test)} samples")
display(styled_table)

#Processing the data
file_path = 'pone.0148733.s001.xlsx'
df = pd.read_excel(file_path, sheet_name='data')

df.columns = df.columns.str.lower()
covariates = ['sex', 'diagnosis', 'location', 'ki', 'gtv', 'stereotactic methods']

df_clean = df.dropna(subset=covariates + ['os', 'status']).copy()
print(f"Dataset is ready with {len(df_clean)} patients after dropping missing values.")

#Creating intervals to categorize the survival time data
bins = [0, 12, 24, 36, 48, 60, 90]
labels = ['0-12 months', '12-24 months', '24-36 months', '36-48 months', '48-60 months', '60-90 months']

df_clean['os_class'] = pd.cut(df_clean['os'], bins=bins, labels=labels, right=True, include_lowest=True)

#This step is in the event that this code be used on a larger dataset in the future
df_clean.dropna(subset=['os_class'], inplace=True)
print(f"Data is categorized into {len(labels)} survival classes. Total patients is now: {len(df_clean)}.")

X = df_clean[covariates]
y = df_clean['os_class']

#Encoding the diagnosis data
le = LabelEncoder()
X['diagnosis'] = le.fit_transform(X['diagnosis'])


#Splitting data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

#Cramer's V evaluation function
def safe_cramers_v(y_true, y_pred, smoothing=1e-6):
    """
    Calculates Cramér's V for categorical association safely,
    even when expected frequencies contain zeros.
    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    confusion_mat = confusion_mat + smoothing

    try:
        chi2 = chi2_contingency(confusion_mat)[0]
    except ValueError:
        #Fall back for chi-squared (included for some versions where it may be necessary)
        expected = np.outer(confusion_mat.sum(axis=1),
                            confusion_mat.sum(axis=0)) / confusion_mat.sum()
        mask = expected > 0
        chi2 = (((confusion_mat - expected) ** 2) / expected)[mask].sum()

    n = confusion_mat.sum()
    phi2 = chi2 / n
    r, k = confusion_mat.shape
    v = np.sqrt(phi2 / min(r - 1, k - 1)) if min(r - 1, k - 1) > 0 else 0
    return v

#Training the new RSF for the categorical times to actually be able to use Cramer's V
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)

rf_cramers_v = safe_cramers_v(y_test, rf_pred)

print(f"  - Cramér's V: {rf_cramers_v:.4f}")

#Above but for XGBoost
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_classifier = XGBClassifier(
    objective='multi:softmax',
    n_estimators=100,
    max_depth=8,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_classifier.fit(X_train, y_train_encoded)
xgb_pred_encoded = xgb_classifier.predict(X_test)

xgb_cramers_v = safe_cramers_v(y_test_encoded, xgb_pred_encoded)

print(f"  - Cramér's V: {xgb_cramers_v:.4f}")
