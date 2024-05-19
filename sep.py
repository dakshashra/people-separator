import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import os

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
gargi_image = face_recognition.load_image_file("gargi.jpg")
gargi_face_encoding = face_recognition.face_encodings(gargi_image)[0]

# Load a second sample picture and learn how to recognize it.
muskan_image = face_recognition.load_image_file("muskan.jpg")
muskan_face_encoding = face_recognition.face_encodings(muskan_image)[0]

tahmina_image = face_recognition.load_image_file("tahmina.jpg")
tahmina_face_encoding = face_recognition.face_encodings(tahmina_image)[0]

kat_image = face_recognition.load_image_file("kat.jpg")
kat_face_encoding = face_recognition.face_encodings(kat_image)[0]

parth_image = face_recognition.load_image_file("parth.jpg")
parth_face_encoding = face_recognition.face_encodings(parth_image)[0]


bd_image = face_recognition.load_image_file("bigdawg.jpg")
bd_face_encoding = face_recognition.face_encodings(bd_image)[0]

nishan_image = face_recognition.load_image_file("nishan.jpg")
nishan_image_face_encoding = face_recognition.face_encodings(nishan_image)[0]

bola_image = face_recognition.load_image_file("bola.jpg")
bola_image_face_encoding = face_recognition.face_encodings(bola_image)[0]

beth_image = face_recognition.load_image_file("beth.jpg")
beth_face_encoding = face_recognition.face_encodings(beth_image)[0]

masood_image = face_recognition.load_image_file("masood.jpg")
masood_face_encoding = face_recognition.face_encodings(masood_image)[0]

mary_image = face_recognition.load_image_file("mary.jpg")
mary_face_encoding = face_recognition.face_encodings(mary_image)[0]

roda_image = face_recognition.load_image_file("roda.jpg")
roda_face_encoding = face_recognition.face_encodings(roda_image)[0]

nishka_image = face_recognition.load_image_file("nishka.jpg")
nishka_face_encoding = face_recognition.face_encodings(nishka_image)[0]

uriel_image = face_recognition.load_image_file("uriel.jpg")
uriel_face_encoding = face_recognition.face_encodings(uriel_image)[0]

aqsa_image = face_recognition.load_image_file("aqsa.jpg")
aqsa_face_encoding = face_recognition.face_encodings(aqsa_image)[0]

harini_image = face_recognition.load_image_file("harini.jpg")
harini_face_encoding = face_recognition.face_encodings(harini_image)[0]

teddy_image = face_recognition.load_image_file("teddy.jpg")
teddy_face_encoding = face_recognition.face_encodings(teddy_image)[0]

emanuel_image = face_recognition.load_image_file("emanuel.jpg")
emanuel_face_encoding = face_recognition.face_encodings(emanuel_image)[0]

botre_image = face_recognition.load_image_file("botre.jpg")
botre_face_encoding = face_recognition.face_encodings(botre_image)[0]

me_image = face_recognition.load_image_file("me.jpg")
me_face_encoding = face_recognition.face_encodings(me_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    gargi_face_encoding,
    muskan_face_encoding,
    tahmina_face_encoding,
    kat_face_encoding,
    parth_face_encoding,
    bd_face_encoding,
    nishan_image_face_encoding,
    bola_image_face_encoding,
    beth_face_encoding,
    masood_face_encoding,
    mary_face_encoding,
    roda_face_encoding,
    nishka_face_encoding,
    uriel_face_encoding,
    aqsa_face_encoding,
    harini_face_encoding,
    teddy_face_encoding,
    emanuel_face_encoding,
    botre_face_encoding,
    me_face_encoding
]
known_face_names = [
    "Gargi",
    "Muskan",
    "Tahmina",
    "Kat",
    "Parth Bhatia",
    "Big Dawg",
    "Nishan",
    "Bolaji",
    "Beth",
    "Masoud",
    "Mary",
    "Rhoda",
    "Nishka",
    "Uriel",
    "Aqsa",
    "Harini",
    "Teddy",
    "Emmanuel",
    "Parth Botre",
    "Daksh"
]
print('Learned encoding for', len(known_face_encodings), 'images.')
indices = [0]* len(known_face_encodings)
for i in range(90):
    # Load an image with an unknown face
    imgdir = "./imgsamples/" + str(i) + ".jpg"
    unknown_image = face_recognition.load_image_file(imgdir)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    print_img = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            indices[best_match_index]+=1
            dirpath = "./"+name
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            print(print_img)
            img_path = dirpath + "/" + str(indices[best_match_index]) + ".jpg"
            print_img.save(img_path)

        
        
        
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width= 20
        text_height= 20

        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    