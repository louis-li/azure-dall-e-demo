import json, os
from azureml.core import Workspace
import base64, io
from PIL import Image
import numpy as np
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def authenticate_client(endpoint, key):
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client

def entity_linking(client, text):

    try:
        result = client.recognize_linked_entities(documents = [text])[0]

        print("Linked Entities:\n")
        for entity in result.entities:
            print("\tName: ", entity.name, "\tId: ", entity.data_source_entity_id, "\tUrl: ", entity.url,
            "\n\tData Source: ", entity.data_source)
            print("\tMatches:")
            for match in entity.matches:
                print("\t\tText:", match.text)
                # print("\t\tConfidence Score: {0:.2f}".format(match.confidence_score))
                # print("\t\tOffset: {}".format(match.offset))
                # print("\t\tLength: {}".format(match.length))
            
    except Exception as err:
        print("Encountered exception. {}".format(err))

def key_phrase_extraction(client, text):

    try:
        response = client.extract_key_phrases(documents = [text])[0]

        if not response.is_error:
            print("Key Phrases:")
            for phrase in response.key_phrases:
                print("\t", phrase)
        else:
            print(response.id, response.error)

    except Exception as err:
        print("Encountered exception. {}".format(err))

def draw_image(text, dimension = 1):
    # print("Recognized: {}".format(text))
    workspace = Workspace.from_config()
    dimension = 1

    # workspace = Workspace.from_config()
    service = workspace.webservices['dall-e-endpoint']

    sample_input = json.dumps({
        'text': text,
        'num_images':dimension * dimension
    })
    response = service.run(sample_input)
    returnedMsg = json.loads(response)['generatedImgs']

    final_image = np.zeros((256*dimension, 256*dimension,3 ))
    for idx,img in enumerate(returnedMsg):
        image = Image.open(io.BytesIO(base64.b64decode(img)))
        row_id = idx // dimension
        col_id = idx % dimension
        final_image[256*row_id:256*(row_id+1), 256*col_id:256*(col_id+1)]=np.array(image)
    Image.fromarray(np.uint8(final_image)).show()
