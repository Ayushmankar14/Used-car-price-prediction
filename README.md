🚗 Car Price Predictor
An interactive Streamlit web app that predicts the selling price of used cars based on user inputs like age, kilometers driven, fuel type, and more. Powered by a Linear Regression model trained on real-world car listings from multiple datasets.


“Get real-time price estimation with just a few clicks!”

🔍 Features
🎯 Predicts used car prices based on key features

💻 Built with Streamlit, scikit-learn, and pandas

🎨 Elegant UI with a background image and responsive design

🧠 Model trained on data from CarDekho, Kaggle, and others

💾 Saves and loads model using pickle for fast predictions

📦 Tech Stack
Tool	Purpose
Python	Core Programming Language
Streamlit	Interactive Web UI
Scikit-learn	Model Training (Linear Regression)
Pandas, NumPy	Data Manipulation & Encoding
Pickle	Model Serialization

🛠️ Installation
Clone the repo

bash
Copy
Edit
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
📁 Project Structure
graphql
Copy
Edit
├── app.py                # Streamlit UI
├── brain.py              # Predict function (used by app or testing)
├── train_model.py        # Model training script
├── car_price_model.pkl   # Trained model
├── car_features.pkl      # Feature columns used in model
├── requirements.txt      # Dependencies
├── Procfile              # For deployment (Heroku/Render)
├── README.md             # This file 😄
└── [datasets]            # Raw CSVs used for training
🚀 Deployment
You can deploy this on Heroku, Render, Hugging Face Spaces, or Streamlit Cloud. Here's a one-line command to deploy on Heroku:

bash
Copy
Edit
heroku create your-car-app && git push heroku main
📊 Example Prediction
Feature	Input
Year	2015
Present Price	₹ 6.5 Lakhs
Kilometers Driven	45,000
Fuel Type	Diesel
Seller Type	Dealer
Transmission	Manual
Owner	First Owner

➡️ Estimated Selling Price: ₹ 4.85 Lakhs

🙋‍♂️ Author
Ayush Kar
📫 [YourEmail@example.com]
🌐 LinkedIn | Portfolio

⭐ Credits
Datasets from Kaggle, CarDekho

Inspired by open-source ML deployment projects

Background image from Unsplash