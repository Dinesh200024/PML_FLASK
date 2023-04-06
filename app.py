from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model2.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    spx = float(request.form['spx'])
    gld = float(request.form['gld'])
    uso = float(request.form['uso'])
    slv = float(request.form['slv'])

    # Create a dataframe with the input features
    input_df = pd.DataFrame({'SPX': [spx], 'GLD': [gld], 'USO': [uso], 'SLV': [slv]})

    # Make a prediction using the loaded model
    prediction = model.predict(input_df)
    result = prediction

    # Render the results page with the prediction
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
