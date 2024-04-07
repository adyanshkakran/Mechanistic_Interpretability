from sklearn.linear_model import LogisticRegression
# import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

class Probe:
    def __init__(self, model_name='logistic'):
        if model_name == 'lr':
            self.model = LogisticRegression()
        elif model_name == 'tree':
            self.model = DecisionTreeClassifier()
        elif model_name == 'mlp':
            self.model = MLPClassifier((64,))
        elif model_name == 'svm':
            self.model = SVC(kernel='linear')
        else:
            raise ValueError('Invalid model_name')

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        print(classification_report(y, y_pred))
        return classification_report(y, y_pred)
    
    def probe(self, X_train, y_train, X_val, y_val, pca=False):
        if pca:
            pca = PCA(n_components=2)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
        self.fit(X_train, y_train)
        return self.evaluate(X_val, y_val)
    
    def plot_decision_boundary(self, X, y):
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Decision Boundary')

def probe_all_models(X_train, y_train, X_val, y_val, pca=False):
    res = []
    models = ['lr', 'tree', 'mlp', 'svm']
    for model in models:
        print(f'Probing {model}')
        probe = Probe(model)
        cr = probe.probe(X_train, y_train, X_val, y_val, pca)
        res.append({
            'model': model,
            'cr': cr
        })
        print('|==============================|')
    return res