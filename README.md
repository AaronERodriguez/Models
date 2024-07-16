# Flask Website for Cyberbullying Classification

This repository contains a Flask website that leverages a custom-made Tensorflow model to classify Twitter posts as either cyberbullying or not. Users can upload text data, and the model will provide predictions based on the input.


![Website Image](./assets/websitePreview.png)

## Getting Started

1. Clone this repository to your local machine.
2. Install the necessary dependencies by running:

pip install -r requirements.txt

3. Run the Flask app:

python index.py

4. Access the website at `http://127.0.0.1:8000/` once you have initiated the website locally.

## How It Works

1. Users upload a text file containing Twitter posts.
2. The Flask app processes the uploaded file and feeds it to the pre-trained Tensorflow model.
3. The model predicts whether each post is cyberbullying or not.
4. The results are displayed on the website.

## File Structure

- `requirements.txt`: Lists the required Python packages.
- `templates/index.html`: HTML template for website
- `app.py`: Main Flask application.
- `tmp/cyber.keras`: Pre-trained Tensorflow model.
- `cyberbullying_tweets.csv`: CSV file for all the data about the Tweets with their labels
- `cyberSecurity.py`: Python file used to create and train the model
- `testingModel.py`: Python terminal app used to write to the terminal and get the resutls
- `views.py`: Flask views for routing

## Contributing

Feel free to contribute to this project by improving the model, enhancing the website, or adding more features!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
