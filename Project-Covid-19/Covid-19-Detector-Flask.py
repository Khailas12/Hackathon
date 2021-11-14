from flask import *
from tensorflow.keras.models import load_model
import cv2
import numpy as np


app = Flask(__name__)


def covid_prediction(image_test):
    model = load_model("model.h5")
    model.compile(optimizer="adam", loss="bianry_crossentropy",
                  metrics=["accuracy"])
    image = cv2.imread(image_test)
    image = cv2.resize(image, (64, 64))
    image = np.reshape(image, [1, 64, 64, 3])

    all_classes = model.predict_classes(image)
    label = ["POSITIVE", "NEGATIVE"]
    return label[all_classes[0][0]]


@app.route("/covid")
def index():
    return render_template("covid-19-index.html")


@app.route("/covid/upload-image", methods=['GET', 'POST'])
def image_upload():
    file = request.files['image']
    if request.method == "POST":
        if file:
            file.save(file.filename)
            label = covid_prediction(file.filename)
            return render_template("covid-19-result.html", name=label)
        else:
            return redirect('/covid')


if __name__ == "__main__":
    app.run(debug=True)