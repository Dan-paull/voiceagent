import torchaudio
import torch
from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
import speech_recognition as sr

model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")

def transcribe(audio_file):
    speech, _ = torchaudio.load(audio_file, normalize=True)
    input_dict = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
    transcripts = model.generate(input_ids=input_dict.input_values, 
                                 attention_mask=input_dict.attention_mask, 
                                 forced_bos_token_id=processor.get_vocab()["<s>"])
    return processor.decode(transcripts[0])

def listen_for_wake_word(wake_word):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        prompt = ''
        try:
            transcription = recognizer.recognize_google(audio)
            print(transcription)
            if wake_word.lower() in transcription.lower():
                
                modified_transcription = transcription.replace(wake_word, '', 1)
                prompt = modified_transcription.strip()
                return True, prompt
            else:
                return "end", prompt
        except sr.UnknownValueError:
            return False, prompt
        

def voice_assistant(wake_word):
        print("Listening...")
        while True:
            wake, acton = listen_for_wake_word(wake_word)
            if wake:
                print("Wake word detected, proceeding to transcribe...")
                input = acton
                print(input)

voice_assistant("action")
