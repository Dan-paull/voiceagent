import os
from llama_index.tools import FunctionTool
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

import webbrowser


import listen
import config

APPLICATION_PATH = '/System/Applications/'


def open_application(application: str) -> str:
    """Used to open an application given the application name"""
    os.system(f"open {APPLICATION_PATH}{application}.app")

    return 'Done'

open_application_tool = FunctionTool.from_defaults(fn=open_application)

website_dictionary = {
    "google": "https://www.google.com",
    "openai": "https://chat.openai.com",
    "prime": "https://www.primevideo.com",
    "netflix": "https://www.netflix.com/browse",
    "phind": "https://www.phind.com",
    "youtube": "https://www.youtube.com",
    "google colab": "https://colab.research.google.com",
    "google drive": "https://drive.google.com",
    "hugging face": "https://huggingface.co"
}

def open_website(website: str) -> str:
    """This function is used to open a website given the website name from the prompt"""

    try:
        try:
            webbrowser.open_new(website_dictionary[website.lower()])
        except:     
            webbrowser.open_new(website.lower())
    except:
        pass

    return website

open_website_tool = FunctionTool.from_defaults(fn=open_website)


llm = OpenAI(model="gpt-3.5-turbo-16k")
explore_agent = OpenAIAgent.from_tools([open_website_tool], llm=llm, verbose=True, max_function_calls= 5)


def voice_assistant(wake_word):
        print("Listening...")
        while True:
            wake, action = listen.listen_for_wake_word(wake_word)
            if wake:
                print("Wake word detected, proceeding to transcribe...")
                computer_response = str(explore_agent.chat(action))
                print(computer_response)

voice_assistant("action")

