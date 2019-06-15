import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import cv2

import zernike
import util
import plotter

UPLOAD_FOLDER = 'uploads'
RESULTS = 'results'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS'] = RESULTS

maybe = {}
zernike_database = util.load_obj('zernike_database_icon_10')
images = util.load_images("icon_sample")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            run_image(filename)
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULTS'],filename)

def run_image(filename):
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    z = zernike.create_query(img)
    res = zernike.test_query(z, zernike_database)
    res = sorted(res, key = lambda tup: tup[1], reverse = True )
    plotter.plot_results(img, res, images, os.path.join(app.config['RESULTS'], filename))