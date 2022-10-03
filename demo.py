import json, os
from azureml.core import Workspace
import base64, io
from PIL import Image
import numpy as np
import azure.cognitiveservices.speech as speechsdk
from utils import authenticate_client, entity_linking, key_phrase_extraction, draw_image
import warnings

warnings.filterwarnings("ignore")
#performs zero-shot speech recognition from the default microphone
# <SpeechRecognitionWithMicrophone>
speech_config = speechsdk.SpeechConfig(subscription=os.environ["SPEECH_API_KEY"], region="eastus")
# The default language is "en-us".
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

#Steup client for Text Analytics
client = authenticate_client(endpoint = os.environ["COGSVC_ENDPOINT_MP"], 
                                key = os.environ["COGSVC_API_KEY_MP"]
                            )   

while True:
    print("0 - Exit")
    print("1 - Speak")

    num = int(input())
    if num==0:
        break
    print("Say something...")
    result = speech_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        key_phrase_extraction(client, result.text)
        entity_linking(client, result.text)
        draw_image(result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
        