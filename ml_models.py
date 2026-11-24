"""
ML Model Adapters
-----------------
Scikit-learn compatible XGBoost classifier
for Smart Electricity Monitoring and Appliance Identification.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class XGBoostApplianceClassifier(ClassifierMixin, BaseEstimator):
    """
    XGBoost wrapper supporting string-based appliance labels.

    Compatible with:
    - Scikit-learn
    - XGBoost 3.x
    - GridSearchCV
    - Cross-validation
    - Joblib serialization
    """

    def __init__(
        self,
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective=None,
        eval_metric=None
    ):

        # Model parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method

        # XGBoost configuration
        self.objective = objective
        self.eval_metric = eval_metric

    def fit(self, X, y):
        """
        Train the XGBoost classifier.
        """

        from xgboost import XGBClassifier

        # Encode appliance labels
        self.label_encoder_ = LabelEncoder()

        encoded_y = self.label_encoder_.fit_transform(y)

        number_of_classes = len(self.label_encoder_.classes_)

        self.n_classes_ = number_of_classes

        # Determine classification objective
        if self.objective is not None:

            objective = self.objective

        elif number_of_classes > 2:

            objective = "multi:softprob"

        else:

            objective = "binary:logistic"

        # Determine evaluation metric
        if self.eval_metric is not None:

            eval_metric = self.eval_metric

        elif number_of_classes > 2:

            eval_metric = "mlogloss"

        else:

            eval_metric = "logloss"

        # Initialize XGBoost model
        self.model_ = XGBClassifier(

            n_estimators=self.n_estimators,

            max_depth=self.max_depth,

            learning_rate=self.learning_rate,

            subsample=self.subsample,

            colsample_bytree=self.colsample_bytree,

            min_child_weight=self.min_child_weight,

            gamma=self.gamma,

            reg_alpha=self.reg_alpha,

            reg_lambda=self.reg_lambda,

            random_state=self.random_state,

            n_jobs=self.n_jobs,

            tree_method=self.tree_method,

            objective=objective,

            eval_metric=eval_metric
        )

        # Train the model
        self.model_.fit(X, encoded_y)

        # Store fitted classifier information
        self.classes_ = self.label_encoder_.classes_

        self.n_features_in_ = self.model_.n_features_in_

        if hasattr(X, "columns"):

            self.feature_names_in_ = np.asarray(
                X.columns,
                dtype=object
            )

        return self

    def predict(self, X):
        """
        Predict appliance classes.
        """

        encoded_predictions = self.model_.predict(X)

        encoded_predictions = np.asarray(
            encoded_predictions,
            dtype=int
        )

        predictions = self.label_encoder_.inverse_transform(
            encoded_predictions
        )

        return predictions

    def predict_proba(self, X):
        """
        Return class probabilities.
        """

        return self.model_.predict_proba(X)

    def get_booster(self):
        """
        Return the underlying XGBoost booster.
        """

        return self.model_.get_booster()