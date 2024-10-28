import requests

url = 'http://localhost:9696/predict'

client_id = 'xyz123'
client = {"job": "management", "duration": 400, "poutcome": "success"}

response = requests.post(url, json=client).json()

print(response)