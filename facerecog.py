import face_recognition

def compare_face(image1, image2):
    print("---- Lets compare Faces using face_recognition lib -----")
    # Load CNH.jpeg and detect faces
    image = face_recognition.load_image_file(image1)
    face_locations = face_recognition.face_locations(image)
    # Get the single face encoding out of CNH
    face_location = face_locations[0]  # Only use the first detected face
    face_encodings = face_recognition.face_encodings(image, [face_location])
    ricardo_known_face_encoding_1 = face_encodings[0]  # Pull out the one returned face encoding

    print("Loaded: ", image1)

    # Load the image with unknown to compare
    image = face_recognition.load_image_file(image2)  # Load the image we are comparing
    unknown_face_encodings = face_recognition.face_encodings(image)
    print("Loaded: ", image2)
    # The known face encodings (can be only 1 - less is faster)
    matches = face_recognition.compare_faces([ricardo_known_face_encoding_1], unknown_face_encodings[0])

    #return True or False if faces recognized.
    return matches[0]

def main():
    image1 = "CNH.jpeg"
    image2 = "Executive.png"
    #image2 = "tom.png"
    matches = compare_face(image1, image2)

    print('Matches for faces in images loaded: ', matches)


if __name__ == "__main__":
    main()
