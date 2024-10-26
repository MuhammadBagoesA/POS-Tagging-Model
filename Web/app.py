from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("../Model/pos_tagger_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/spok", methods=["GET", "POST"])
def spok():
    if request.method == "POST":
        sentence = request.form["sentence"]
        tagged_sentence = tag_sentence(model, sentence)
        return render_template("spok.html", tagged_sentence=tagged_sentence)
    return render_template("spok.html")

def tag_sentence(tagger, sentence):
    tokens = sentence.split()
    return tagger.tag(tokens)

if __name__ == "__main__":
    app.run(debug=True)
