from flask import Flask, request, jsonify
import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        X = np.insert(X, 0, 1, axis=1)
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

app = Flask(__name__)
perceptron = Perceptron(input_size=2) 

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    X = np.array(data['X'])
    y = np.array(data['y'])
    perceptron.fit(X, y)
    return jsonify({"message": "Model trained!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array(data['X'])
    X = np.insert(X, 0, 1)  
    prediction = perceptron.predict(X)
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
