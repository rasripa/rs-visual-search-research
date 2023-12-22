import requests
import json
import uuid
import base64
from PIL import Image

class ImageSearchService:
    def __init__(self):
        self.api_url = 'https://inht4uuc7k.execute-api.us-west-2.amazonaws.com/dev'
        self.jobs = {}

    
    def resize_image(photo, width, height):
        Image.MAX_IMAGE_PIXELS = 100000000
        with Image.open(photo) as image:
            image.verify()
        with Image.open(photo) as image:
            if image.format in ["JPEG", "PNG"]:
                image.thumbnail((width, height))
        return image
      
    def search_image(self, uploaded_image):
        jobId = str(uuid.uuid4())
        self.jobs[jobId] = {'jobStatus': 'init'}
        resized_image = self.resize_image(uploaded_image,2048,2048)
        base64Image = base64.b64encode(resized_image.read()).decode('utf-8')
        self.jobs[jobId]['image'] = base64Image
        payload = {'image': base64Image}
        

