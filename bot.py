import json
import requests

FACEBOOK_GRAPH_URL = 'https://graph.facebook.com/v2.6/me/messages'


class Bot(object):
    def __init__(self, access_token, api_url=FACEBOOK_GRAPH_URL):
        self.access_token = access_token
        self.api_url = api_url

    def send_text_message(self, psid, message, messaging_type="RESPONSE"):
        headers = {
            'Content-Type': 'application/json'
        }

        data = {
            'messaging_type': messaging_type,
            'recipient': {'id': psid},
            'message': {'text': message}
        }

        params = {'access_token': self.access_token}
        response = requests.post(self.api_url,
                                 headers=headers, params=params,
                                 data=json.dumps(data))
        print (response.content)
