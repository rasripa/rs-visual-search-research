import traceback
import websockets
import asyncio
import requests
import json
import uuid
import concurrent.futures
import threading
import faiss
import numpy as np
from collections import defaultdict
import aiohttp
import aiojobs
import backoff
from requests.exceptions import ConnectionError,ReadTimeout
import random

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )

class ImageSearchService:
    def __init__(self):
        self.ws_url = 'wss://oc4duy013j.execute-api.us-west-2.amazonaws.com/production'
        self.jobs = {}
        self.connection = None
        self.clientId = str(uuid.uuid4())
        self.listener_thread = threading.Thread(target=self.start_listener)
        self.listener_thread.start()
        self.text_index = faiss.read_index("text.index")
        self.image_index = faiss.read_index("image.index")
        self.scheduler = asyncio.run(aiojobs.create_scheduler())
        try:
            with open("global_image_id_to_record.json", "r") as file:
                self.image_id_to_record = json.load(file)
            with open("global_text_id_to_record.json", "r") as file:
                self.text_id_to_record = json.load(file)
        except Exception as e:
            print(f"Error loading JSON: {str(e)}")


        
    def normalise_embedding(self,embedding):
        return embedding / np.linalg.norm(embedding) 
     
    async def search_embedding(self,embedding, job_id, k=5):
        normalised_embedding = self.normalise_embedding(np.array(embedding))
        
        # Search in Faiss indexes
        image_distances, image_indices = self.image_index.search(normalised_embedding.reshape(1, -1), k)
        text_distances, text_indices = self.text_index.search(normalised_embedding.reshape(1, -1), k)
        print(image_distances, image_indices)
        print(text_distances, text_indices)
        #retrieve product ids
        image_product_ids = [self.image_id_to_record[str(i)] for i in image_indices[0]]
        try:
            text_product_ids = [self.text_id_to_record[str(i)] for i in text_indices[0]]
        except Exception as e:
            print("error is here")
            print((text_distances, text_indices))
            raise e
    
        # Initialize a scoring dictionary
        score_dict = defaultdict(float)
        
        # Weighted scoring
        image_weight = .2
        text_weight = .8
    
        for i, product_id in enumerate(image_product_ids):
            score_dict[product_id] += image_distances[0][i] * image_weight
            
        for i, product_id in enumerate(text_product_ids):
            score_dict[product_id] += text_distances[0][i] * text_weight
    
        # Sort by score and take top k results
        sorted_results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_results = [item[0] for item in sorted_results[:k]]
        
        self.jobs[job_id].update({'jobStatus': 'Search Complete', 'top_results': top_results})
        
        return top_results
            
    def start_listener(self):
        asyncio.run(self.connect())

    async def connect(self):
        while True:
            try:
                self.session = aiohttp.ClientSession()
                self.connection = await self.session.ws_connect(f"{self.ws_url}/?clientId={self.clientId}")
                await self.listen_for_messages()
            except (aiohttp.ClientConnectorError, aiohttp.WSServerHandshakeError) as e:
                print(f"Connection error: {str(e)}, retrying in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Unhandled error: {str(e)}, retrying in 5 seconds...")
                await asyncio.sleep(5)

    async def submit_job(self, uploaded_file):
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {'jobStatus': 'submitting Job','uploaded_file': uploaded_file}
        message = {
            'action': 'submitJob',
            'jobId': job_id,
            'clientId': self.clientId
        }
        await self.send_message(message)
        
    async def get_products(self, job_id):
        message = {
            'action': 'getProducts',
            'jobId': job_id,
            'clientId': self.clientId,
            'productIds': self.jobs[job_id]['top_results']
        }
        await self.send_message(message)

    async def get_presigned_url(self, job_id):
        message = {
            'action': 'generatePresignURL',
            'jobId': job_id,
            'clientId': self.clientId,
            'fileType': self.jobs[job_id]['uploaded_file'].type
        }
        await self.send_message(message)

    @backoff.on_exception(
        backoff.expo,
        (ConnectionError, ReadTimeout),
        max_time=random.randint(500, 1000),
        on_backoff=backoff_hdlr,
        giveup=lambda e: 'ConnectionError' not in str(e) and 'ReadTimeout' not in str(e)
    ) 
    async def upload_image(self, presigned_url, uploaded_file, job_id):
        try:
            print("Uploading image")
            headers = {'Content-Type': uploaded_file.type.split('/')[1]}
            response = requests.put(presigned_url, data=uploaded_file.read(), headers=headers)
            response.raise_for_status()
            if response.status_code == 200:
                print("Image uploaded successfully")
                self.jobs[job_id].update({'jobStatus': 'Image Uploaded'})
                try:
                    self.get_embedding_sync(job_id)
                except Exception as e:
                    print(f"Error getting embedding: {str(e)}")
        except Exception as e:
            print(f"Error uploading image: {str(e)}")

    async def get_embedding(self, job_id):
        message = {
            'action': 'getEmbedding',
            'jobId': job_id,
            'clientId': self.clientId,
            'fileType': self.jobs[job_id]['uploaded_file'].type.split('/')[1]
        }
        await self.send_message(message)

    async def send_message(self, message):
        if not self.connection or self.connection.closed:
            print("WebSocket is disconnected, attempting to reconnect...")
            await self.scheduler.spawn(self.connect())
        try:
            print(f"Sending message: {json.dumps(message)}")
            await self.connection.send_json(message)
        except Exception as e:
            print(f"Error sending message: {str(e)}")

    async def listen_for_messages(self):
        try:
            async for message in self.connection:
                print(f"Received message: {message.data}")
                self.handle_message(json.loads(message.data))
        except KeyError as e:
            print(f"KeyError: {str(e)}")
            traceback.print_exc()
        except Exception as e:
            print(f"Error listening for messages: {type(e).__name__}: {str(e)}")
            traceback.print_exc()


    def handle_message(self, message):
        job_id = message.get('jobId')
        if 'jobStatus' in message:
            self.jobs[job_id].update({'jobStatus': message['jobStatus']})
            if message['jobStatus'] == 'SUBMITTED':
                self.get_presigned_url_sync(job_id)
        if 'presigned_url' in message:
            self.jobs[job_id].update({
                'presigned_url': message['presigned_url'],
                'jobStatus': 'PreSignURLGen'
            })
            self.upload_image_sync(message['presigned_url'], job_id)
        if 'embeddingResult' in message:
            self.jobs[job_id].update({
                'embeddingResult': message['embeddingResult'],
                'jobStatus': 'Image Embedding Computed'
            })
            try:
                self.search_embedding_sync(message['embeddingResult'], job_id=job_id)
            except Exception as e:
                print(f"Error searching embedding: {str(e)}")
            try:
                self.get_products_sync(job_id)
            except Exception as e:
                print(f"Error getting products: {str(e)}")
        if 'productItems' in message:
            self.jobs[job_id].update({'productItems': message['productItems']})
            self.jobs[job_id].update({'jobStatus': message['jobStatus']})
            

    def run_in_executor(self, coro):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    def submit_job_sync(self, uploaded_file):
        return self.run_in_executor(self.submit_job(uploaded_file))
    
    def get_presigned_url_sync(self, job_id):
        return self.run_in_executor(self.get_presigned_url(job_id))

    def upload_image_sync(self, presigned_url, job_id):
        uploaded_file = self.jobs[job_id]['uploaded_file']
        return self.run_in_executor(self.upload_image(presigned_url, uploaded_file, job_id))

    def get_embedding_sync(self, job_id):
        return self.run_in_executor(self.get_embedding(job_id))
    
    def search_embedding_sync(self,embedding, job_id):
        return self.run_in_executor(self.search_embedding(embedding,job_id=job_id))
    
    def get_products_sync(self, job_id):
        return self.run_in_executor(self.get_products(job_id))
