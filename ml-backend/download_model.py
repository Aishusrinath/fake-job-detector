import requests
 
# direct download link
url = "https://drive.google.com/file/d/15Vqn8-rAIIQhEdxYUcgwx1jC6znrmXx_/view?usp=sharing"
 
output = "email_image_model.pt"
 
print("Downloading model...")
r = requests.get(url)
 
with open(output, "wb") as f:
    f.write(r.content)
 
print("Model downloaded successfully!")
