import requests
import json
import uuid
import base64
from PIL import Image
from io import BytesIO



class ImageSearchService:
    def __init__(self):
        self.api_url = 'https://1iydovg8jf.execute-api.us-west-2.amazonaws.com/dev'
        self.jobs = {}

    
    def resize_image(self, image, width, height):
        Image.MAX_IMAGE_PIXELS = 100000000
        photo = Image.open(image)
        photo.thumbnail((width, height))
        return photo
        
        
    def search_image(self, uploaded_image):
        jobId = str(uuid.uuid4())
        self.jobs[jobId] = {'jobStatus': 'init'}
        resizedImage = self.resize_image(uploaded_image,2048, 2048)
        buffer = BytesIO()
        resizedImage.save(buffer, format=resizedImage.format)
        base64Image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.jobs[jobId]['image'] = base64Image
        payload = {'image': base64Image}
        headers = {'content-type': 'application/json'}
        response = requests.post(f"{self.api_url}/search", data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            try: 
                self.jobs[jobId].update({'jobStatus': 'Search Complete'})
                result = response.json()
            except ValueError:
                print("empty response")
                return None
            productItems = json.loads(result['body'])['products']['ProductTable']
            self.jobs[jobId].update({'productItems': productItems})
            return productItems
        else:
            print(f"Error searching image: {response.status_code}")
            return None
        

