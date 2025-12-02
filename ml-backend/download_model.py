import requests
 
# direct download link
url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
 
output = "email_image_model.pt"
 
print("Downloading model...")
r = requests.get(url)
 
with open(output, "wb") as f:
    f.write(r.content)
 
print("Model downloaded successfully!")
