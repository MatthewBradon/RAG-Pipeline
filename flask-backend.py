from flask import Flask, request, jsonify
from flask_cors import CORS
from RAGPipeline import RAGPipeline
app = Flask(__name__)
CORS(app)
model = RAGPipeline()
@app.route('/chat', methods=['POST'])
def sendMessage():
    data = request.get_json()
    prompt = data['prompt']
    response = {
        'status': 'success',
        'prompt': prompt,
        'response': model.askQuestion(prompt)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    
