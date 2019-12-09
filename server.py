from flask import Flask, render_template, request, jsonify
from model import run_model
import json

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def home():
        return render_template('/index.html')

@app.route("/compute", methods=['POST'])
def compute():
    data = request.files['file']
    data.save('data/input_data.mat')
    return json.dumps({ 'result': run_model() })
    
if __name__ == "__main__":
    app.run(debug=True)
