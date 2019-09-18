import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import cv2

import zernike
import orb
import sift
import combined
import util
import plotter
#comment just to gitdg
UPLOAD_FOLDER = 'uploads'
RESULTS = 'results'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS'] = RESULTS

zernike_database = util.load_obj('zernike_database_icon_50')
orb_database = util.load_obj('orb_database_icon_50')
sift_database = util.load_obj('sift_database_icon_50')
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
    <title>TEST YOUR IMAGE FOR PLAGIARISM!</title>
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

    print("starting")
    o = orb.create_query(img)
    o = orb.test_query(o, orb_database)
    s = sift.create_query(img)
    s = sift.test_query(s, sift_database)
    z = zernike.create_query(img)
    z = zernike.test_query(z, zernike_database)
    res = combined.test_combined([z,o,s], [1,5,5])

    plotter.plot_results(img, res, images, os.path.join(app.config['RESULTS'], filename))