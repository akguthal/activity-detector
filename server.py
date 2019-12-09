from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def home():
        return render_template('/index.html')

@app.route("/compute", methods=['POST'])
def compute():
    data = request.files['file'].read()
    return jsonify({ 'result': data })
    
if __name__ == "__main__":
    app.run(debug=True)
