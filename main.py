from flask import Flask, render_template, request
import vertexai
from vertexai.language_models import GroundingSource
from google.oauth2.service_account import Credentials
import vertexai.language_models
from vertexai.language_models import TextGenerationModel

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['user_input']
    # Do something with the message, like processing or storing it
    print("Received message:", message)

    key_path='vertixai-poc-413616-bf407ba0d2b6.json'
    credentials = Credentials.from_service_account_file(
        key_path)
    PROJECT_ID = 'vertixai-poc-413616'
    REGION = 'us-central1'
    vertexai.init(project = PROJECT_ID, location = REGION,credentials=credentials)
    chat_model = vertexai.language_models.ChatModel.from_pretrained("chat-bison")
    # grounding_source = GroundingSource.VertexAISearch(data_store_id="kec-data_1714063035315", location="global"),

    parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.9,
    "top_p": 1
    }
    grounding_source = GroundingSource.VertexAISearch(data_store_id="kec-web_1714210784959", location="global", project="vertixai-poc-413616")
    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(
        f"""this is the onsite chatbot for Kongu Engineering College, give out the output in proper sentence

input: {message}
output:
""",
        **parameters,
        grounding_source=grounding_source
    )
    print(f"Response from Model: {response.text}")
    return response.text



if __name__ == '__main__':
    app.run(debug=True)