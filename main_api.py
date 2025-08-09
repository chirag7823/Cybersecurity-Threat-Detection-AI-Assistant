import pickle
from flask import Flask, request, jsonify, render_template # ADDED render_template here
import numpy as np
import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import socket
import whois
from datetime import datetime

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Your Pre-trained Model ---
# Make sure to have your model file (e.g., 'phishing_model.pkl') in the same directory.
# This is a placeholder; you MUST replace it with your actual model loading.
try:
    with open(r'C:\Users\student\Desktop\314\API\phishing_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Model file 'phishing_model.pkl' not found. The API will not work without it.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# --- Feature Extraction Functions ---

def get_domain_from_url(url):
    """Extracts the domain name from a URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ''

def get_domain_in_title(url, domain):
    """
    Checks if the domain is present in the page title.
    Returns 1 if present, 0 otherwise.
    """
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.lower() if soup.title and soup.title.string else ''
        return 1 if domain.lower() in title else 0
    except Exception:
        # If the page can't be reached or has no title, assume it's suspicious
        return 1

def get_ratio_digits_url(url):
    """Calculates the ratio of digits to total characters in the URL."""
    digits = sum(c.isdigit() for c in url)
    total_len = len(url)
    return digits / total_len if total_len > 0 else 0

def get_phish_hints(url):
    """Counts the occurrences of common phishing-related keywords in the URL."""
    hints = ['login', 'secure', 'account', 'update', 'verify', 'signin', 'banking']
    count = 0
    for hint in hints:
        if hint in url.lower():
            count += 1
    return count

def get_google_index(url):
    """
    Checks if the URL is indexed by Google.
    Returns 1 if indexed, 0 otherwise.
    Note: This is a simulation and can be unreliable.
    """
    try:
        # A simple way to check is to make a search query for the site
        query = f"site:{url}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(f"https://www.google.com/search?q={query}", headers=headers, timeout=5)
        # If the response doesn't contain certain "no results" phrases, it's likely indexed.
        return 0 if "did not match any documents" in response.text or "no results found" in response.text else 1
    except Exception:
        # If search fails, we can't be sure, so we treat it as not indexed.
        return 0

def get_ip(domain):
    """
    Checks if the domain is an IP address.
    Returns 1 if it is an IP, 0 otherwise.
    """
    # Regex to check for IP address format
    ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    if ip_pattern.match(domain):
        return 1
    # Check if the domain resolves to an IP, which is normal. This feature
    # specifically checks if the domain *itself* is the IP.
    return 0


def get_length_url(url):
    """Returns the total length of the URL."""
    return len(url)

def get_domain_age(domain):
    """
    Calculates the age of the domain in days.
    Returns -1 if the age cannot be determined.
    """
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        # Handle cases where creation_date is a list
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            age = (datetime.now() - creation_date).days
            return age
        return -1
    except Exception:
        # If WHOIS lookup fails, return a value indicating an issue
        return -1

def get_https_token(url):
    """
    Checks if 'https' is part of the domain name itself (a common phishing trick).
    Returns 1 if 'https' is in the domain part, 0 otherwise.
    """
    domain = get_domain_from_url(url)
    return 1 if 'https' in domain.lower() else 0

import requests

def get_page_rank(domain):
    try:
        headers = {
            'API-OPR': 'k0kk44o0ogc8osc8o0sc0ok0oos8oogwsw40sok0'  # Replace with your OpenPageRank API key
        }
        response = requests.get(f"https://openpagerank.com/api/v1.0/getPageRank?domains[]={domain}", headers=headers)
        data = response.json()

        # Check if data is valid
        if "response" in data and len(data["response"]) > 0:
            rank_data = data["response"][0]
            return float(rank_data.get("page_rank_decimal", 1.0))  # Default to 1.0 if missing
        else:
            return 1.0  # Neutral default

    except Exception as e:
        print("Error fetching PageRank:", e)
        return 1.0



# --- API Endpoint ---

# NEW: Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives a URL, extracts features, and returns a prediction."""
    if not model:
        return jsonify({'error': 'Model is not loaded. Cannot make predictions.'}), 500

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL not provided in request body.'}), 400

    url = data['url']
    domain = get_domain_from_url(url)

    if not domain:
        return jsonify({'error': 'Could not parse a valid domain from the URL.'}), 400

    # Extract all features in the correct order for the model
    features = {
        'domain_in_title': get_domain_in_title(url, domain),
        'ratio_digits_url': get_ratio_digits_url(url),
        'phish_hints': get_phish_hints(url),
        'google_index': get_google_index(url),
        'ip': get_ip(domain),
        'length_url': get_length_url(url),
        'domain_age': get_domain_age(domain),
        'https_token': get_https_token(url),
        'page_rank': get_page_rank(domain)
    }

    # The order MUST match the order your model was trained on.
    # ['domain_in_title','ratio_digits_url','phish_hints','google_index','ip', 'length_url', 'domain_age', 'https_token', 'page_rank']
    feature_list = [
        features['domain_in_title'],
        features['ratio_digits_url'],
        features['phish_hints'],
        features['google_index'],
        features['ip'],
        features['length_url'],
        features['domain_age'],
        features['https_token'],
        features['page_rank']
    ]

    # Convert to a numpy array for the model
    final_features = np.array(feature_list).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)
    prediction_proba = model.predict_proba(final_features)

    # Prepare the response
    # Assuming 1 is 'phishing' and 0 is 'legitimate'
    result_label = 'phishing' if prediction[0] == 1 else 'legitimate'
    confidence = float(prediction_proba[0][prediction[0]])

    response = {
        'url': url,
        'prediction': result_label,
        'confidence': f"{confidence:.2f}",
        'features_extracted': features
    }

    return jsonify(response)


# --- Run the App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the API accessible from your network
    app.run(host='0.0.0.0', port=5000, debug=True)