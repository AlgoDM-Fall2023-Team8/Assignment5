import base64
import requests
import boto3

import boto3
import openai
import json

responses = []

# Connect to S3  
s3 = boto3.client('s3', aws_access_key_id='AKIA4CE2PKJDNL4H545R', aws_secret_access_key='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')
bucket_name = 'anarsak1'

# Connect to ChatGPT API
openai.api_key = "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"

# List images in S3 bucket
images = s3.list_objects(Bucket=bucket_name)['Contents']  


def form_and_table_understanding(image_path, prompt_text,key):
  """ form_and_table_understanding """
  base64_image = image_path # Path to your image
  headers = {"Content-Type": "application/json","Authorization": f"Bearer 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"}
  payload = {
      "model": "gpt-4-vision-preview",
      #"response_format" : { "type": "json_object" },
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt_text
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
  }
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  
  return(response.json(),key)



for image in images:
    key = image['Key']
    
    # Download image from S3
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    image_bytes = obj['Body'].read()
    
    # Base64 encode image
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt_text = "Generate 10 labels of this image  type_of_image : clothes,food ; gender:woman clothes or men ; sleves:shortorlong ; collar:with or without collar ; formal_informal : formal or casual ; colour:red ; age_group : 3 to 4 years,20 to 30 years ; brand_name : nokia,calvin klein ; other_info : more about image"

    

        # Call API
    resp,key = form_and_table_understanding(base64_image, prompt_text, key)
    print(key)
    # Append response to array

    element = {
        'id': key,
        'annotations': resp['choices'][0]['message']['content'],
  

    }
    
   

    responses.append(element)

print(responses)

# Convert the array to JSON format
json_data = json.dumps(responses, indent=2)  # indent for pretty printing, optional

# Save JSON data to a file
with open('data.json', 'w') as file:
    file.write(json_data)

# Function to encode the image
def encode_image(image_path):
  print(image_path)  
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def form_and_table_understanding(image_path, prompt_text):
  """ form_and_table_understanding """
  base64_image = encode_image(image_path)  # Path to your image
  headers = {"Content-Type": "application/json","Authorization": f"Bearer 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"}
  payload = {
      "model": "gpt-4-vision-preview",
      #"response_format" : { "type": "json_object" },
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt_text
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
  }
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return(response.json())


# Path to your image
# image_path = "10054.jpeg"
# prompt_text = "Generate 10 labels of this image  type_of_image : clothes,food ; gender:woman clothes or men ; sleves:shortorlong ; collar:with or without collar ; formal_informal : formal or casual ; colour:red ; age_group : 3 to 4 years,20 to 30 years ; brand_name : nokia,calvin klein ; other_info : more about image"
# resp = form_and_table_understanding(image_path, prompt_text)
# print(resp['choices'][0]['message']['content'])

# import re
# import json

# # Provided information in numbered list format
# numbered_list = resp['choices'][0]['message']['content']
# # Define fixed keys
# keys = [
#     "Type of Image",
#     "Gender",
#     "Sleeves",
#     "Collar",
#     "Formal Informal",
#     "Color",
#     "Age Group",
#     "Brand Name",
#     "Other Info 1"]

# # Split the numbered list into lines
# lines = numbered_list.split('\n')

# # Create a dictionary with fixed keys
# image_info = dict(zip(keys, [line.split(":")[1].strip() for line in lines if line]))

# # Convert to JSON format
# json_data = json.dumps(image_info, indent=2)

# # Print the JSON data
# print(json_data)




