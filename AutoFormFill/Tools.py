import requests
from bs4 import BeautifulSoup

def get_visible_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.extract() 

    visible_text = soup.get_text(separator=' ')


    visible_text = ' '.join(visible_text.split())

    return visible_text

url = 'https://docs.google.com/forms/d/e/1FAIpQLSeCoV7Ul1tWe3OBTCQPaa0LmHfdNYQaIJcQMG-DpMAR6f5wKQ/viewform' 
text = get_visible_text(url)

print(text)
