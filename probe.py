from sklearn.linear_model import LogisticRegression
# import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

class Probe:
    def __init__(self, model_name='logistic'):
        if model_name == 'lr':
            self.model = LogisticRegression()
        elif model_name == 'tree':
            self.model = DecisionTreeClassifier()
        elif model_name == 'mlp':
            self.model = MLPClassifier((128, 64))
        else:
            raise ValueError('Invalid model_name')

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        print(classification_report(y, y_pred))
    
    def probe(self, X_train, y_train, X_val, y_val):
        self.fit(X_train, y_train)
        self.evaluate(X_val, y_val)