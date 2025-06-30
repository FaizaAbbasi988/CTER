import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

"""Splitting the Data into Training & Testing Set"""
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify= y, random_state =0)
# Define the ELMClassifier class
class ELMClassifier:
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden

    def fit(self, X, y):
        # Generate random input weights and biases
        self.input_weights = np.random.randn(X.shape[1], self.n_hidden)
        self.bias = np.random.randn(self.n_hidden)

        # Calculate hidden activations
        hidden_activations = np.dot(X, self.input_weights) + self.bias
        hidden_activations = 1 / (1 + np.exp(-hidden_activations))  # Sigmoid activation

        # Calculate output weights using the pseudo-inverse
        self.output_weights = np.dot(np.linalg.pinv(hidden_activations), y)

    def predict(self, X):
        # Calculate hidden activations for prediction
        hidden_activations = np.dot(X, self.input_weights) + self.bias
        hidden_activations = 1 / (1 + np.exp(-hidden_activations))  # Sigmoid activation

        # Calculate predictions using the output weights
        y_pred = np.dot(hidden_activations, self.output_weights)
        # y_pred = np.argmax(softmax(y_pred), axis=1)
        return y_pred
# Create an ELMClassifier instance with the desired number of hidden neurons
elm_model = ELMClassifier(n_hidden=1000)

# Train the ELM model on the training data
elm_model.fit(X_train, y_train)

# Predict using the ELM model
y_pred_elm = elm_model.predict(X_test)

# Calculate accuracy and other metrics
accuracy_elm = metrics.accuracy_score(y_test, (y_pred_elm > 0.5).astype(int))
# roc_curve_elm = metrics.roc_curve(y_test, y_pred_elm)
confusion_matrix_elm = metrics.confusion_matrix(y_test, (y_pred_elm > 0.5).astype(int))
classification_report_elm = metrics.classification_report(y_test, (y_pred_elm > 0.5).astype(int))

# Print the results
print("ELM Accuracy:", accuracy_elm)
# print("ELM ROC Curve:", roc_curve_elm)
print("ELM Confusion Matrix:\n", confusion_matrix_elm)
print("ELM Classification Report:\n", classification_report_elm)
