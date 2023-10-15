from deepface import DeepFace

result = DeepFace.represent(img_path = "test.jpg")
print(len(result[0]["embedding"]))