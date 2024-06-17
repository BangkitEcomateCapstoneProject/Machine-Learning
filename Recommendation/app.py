from flask import Flask, request, jsonify
from Recommendation import ArticleRecommendation
from flask import Response
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

@app.route("/")
def index():
    """
    Index endpoint to check if the server is running.
    Returns a simple "Hello World" message.
    """
    return "Hello World"

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Recommendation endpoint.
    Expects a JSON payload with a "text" attribute containing the user's input text.
    
    Returns:
        - JSON response with recommended articles if successful.
        - 400 Bad Request error if the "text" attribute is missing in the request.
    """
    input_user = request.get_json(force=True)
    if "text" not in input_user:
        return (
            jsonify(error="Bad Request", message="JSON must contain 'text' attribute"),
            400,
        )
    text = input_user["text"]
    
    # Initialize the article recommendation model
    model = ArticleRecommendation()
    
    # Generate recommendations based on the input text and data path from environment variables
    result = model.recommendation(
        os.environ.get("DATA_PATH"),
        text,
    )
    
    # Return the recommendation result as a JSON response
    return result.to_json(orient="records", lines=True)

if __name__ == "__main__":
    # Run the Flask application in debug mode
    app.run(debug=True)