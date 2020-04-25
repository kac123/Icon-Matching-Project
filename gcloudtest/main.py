"Plot a PNG using matplotlib in a web request, using Flask."

# Install dependencies, preferably in a virtualenv:
#
#     pip install flask matplotlib
#
# Run the development server:
#
#     python app.py
#
# Go to http://localhost:5000/plot.png and see a plot of random data.

import random
import io

from flask import Flask, make_response, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)


@app.route('/plot.png')
def plot():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]

    axis.plot(xs, ys)
    FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)