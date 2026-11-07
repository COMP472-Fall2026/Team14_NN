"""Classifiers. Every model follows the same interface:

    model = SomeClassifier(**hyperparameters)
    model.fit(X_train, y_train)   # returns self
    y_pred = model.predict(X)     # 1-D int array of class labels

so any of them can be passed to ``comp472_project.evaluation.evaluate``.

k-NN, Decision Tree and Random Forest must be implemented from scratch
(NumPy only). The neural network may use torch.nn / torch.optim.
"""

from comp472_project.models.decision_tree import DecisionTreeClassifier
from comp472_project.models.knn import KNNClassifier
from comp472_project.models.neural_net import NeuralNetClassifier
from comp472_project.models.random_forest import RandomForestClassifier

__all__ = [
    "DecisionTreeClassifier",
    "KNNClassifier",
    "NeuralNetClassifier",
    "RandomForestClassifier",
]
