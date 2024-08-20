# flake8: noqa
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
from transformers import AutoTokenizer, MixtralForCausalLM


listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) # for female voice


def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'alexa' in command:
                command = command.replace('alexa', '')
                
    except:
        pass
    return command


def run_alexa():
    command = take_command()
    if 'play' in command:
        song = command.replace('play', '')
        talk('playing ' + song)
        # playing a video or a song on youtube
        pywhatkit.playonyt(song)
        
    elif 'time' in command:
        # fetching and displaying time
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
        
        
    elif 'who is' in command:
        # running on the basic assumption that the user will need to access information on a person in this format only
        person = command.replace('who the heck is', '')
        info = wikipedia.summary(person, 1)
        talk(info)
        
    elif 'joke' in command:
        talk(pyjokes.get_joke())
    else:
        model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

        prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        talk(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

while True:
    run_alexa()