from __future__ import print_function
from flask import Flask, request ,jsonify
import werkzeug
import binascii
from PIL import Image
import numpy as np
import scipy
from scipy.cluster.vq import kmeans
import webcolors
NUM_CLUSTERS = 3

app = Flask(__name__)

@app.route('/upload', methods=["POST"])
def upload():
    if (request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("/home/mariemgalal/images/"+ filename)
        im = Image.open("/home/mariemgalal/images/"+ filename)
        im = im.resize((150, 150))      # optional, to reduce time
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
        index_max = scipy.argmax(counts)                    # find most frequent
        peak = codes[index_max]
        colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
        t=np.int(peak[0])
        r=np.int(peak[1])
        e=np.int(peak[2])
        c=[t,r,e]

        def closest_colour(requested_colour):
            min_colours = {}
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - requested_colour[0]) ** 2
                gd = (g_c - requested_colour[1]) ** 2
                bd = (b_c - requested_colour[2]) ** 2
                min_colours[(rd + gd + bd)] = name
            return min_colours[min(min_colours.keys())]

        def get_colour_name(requested_colour):
            try:
                closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
            except ValueError:
                closest_name = closest_colour(requested_colour)
                actual_name = None
            return actual_name, closest_name

        actual_name, closest_name = get_colour_name(peak)

      
        return jsonify({
            "message": closest_name 
            })