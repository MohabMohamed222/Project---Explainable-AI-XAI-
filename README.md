# Project---Explainable-AI-XAI-
This project aims to gain hands-on experience in implementing and comparing key Explainable AI (XAI) techniques that make complex machine learning models interpretable.


1. CDC Diabetes Health Indicators (Tabular, Classification)
•	Scenario: A public health agency wants to use a machine learning model to predict a person's diabetes risk based on health survey data. They need to understand which health factors are the most significant drivers of a positive diagnosis to plan effective public interventions.

•	Problem (Business): The model is a black box. The agency needs to know, on a population level (global), which factors are most important (e.g., HighBP vs. BMI). They also need to explain to a specific individual (local) why they were flagged as high-risk.

•	Problem (Dataset): This is a large tabular dataset (253,680 rows). The main challenge is that the features are almost entirely binary or categorical (e.g., HighBP = 0 or 1, Age = 1-13 levels). You must OneHotEncode these. The dataset is also imbalanced (86% non-diabetic, 14% diabetic), so you will need to use techniques like SMOTE or class_weight balancing.

Solution (XAI Application):
Global:
* Permutation Feature Importance: Use this to get a definitive ranking of which features (e.g., HighBP, HighChol, BMI, Age) are most critical to the model's accuracy.
* Partial Dependence Plots (PDP): Plot the average probability of diabetes against BMI or Age (the few numerical/ordinal features) to see the trend.
* Individual Conditional Expectation (ICE): Use this to see if the BMI trend is the same for both smokers and non-smokers.
Local:
* LIME (Tabular): Explain a single prediction: "This patient was flagged as high-risk because HighBP = 1, Age = 10, and BMI = 35."
•	Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset






2. NYC Taxi Fare Prediction (Tabular, Regression)
•	Scenario: A ride-sharing company wants to predict taxi fare amounts in real-time. The model must be accurate, but the company also needs to understand what drives fare prices (e.g., distance, time of day, location) to optimize pricing and explain high fares to customers.

•	Problem (Business): The model predicts a continuous value (fare_amount). The company needs to (a) globally understand its pricing logic and (b) locally justify a single, expensive fare to a customer.

•	Problem (Dataset): This is a very large tabular dataset (millions of rows; you will likely use a large sample, e.g., 1-5 million). The main challenge is extensive feature engineering. The raw data (pickup_datetime, pickup_longitude, dropoff_latitude, etc.) is not usable directly. You must create new features like:

*  haversine_distance (from coordinates)

*  hour_of_day, day_of_week, month (from pickup_datetime)

*  distance_to_jfk, distance_to_lga (distance to airports)

Solution (XAI Application):
Global:
*	Permutation Feature Importance: Quantify how important haversine_distance is versus hour_of_day or day_of_week.
*	Partial Dependence Plots (PDP): This is perfect for plotting fare_amount against your engineered features. You can visualize "surge pricing" by plotting fare_amount vs. hour_of_day (you'll see spikes at 8 AM and 6 PM) or fare_amount vs. haversine_distance.
Local:
*	LIME (Tabular): Explain a single high fare: "This $75 fare was high because haversine_distance = 15 miles (+ $40), hour_of_day = 17 (5 PM) (+ $10), and distance_to_jfk = 2 miles (+ $15)."
•	Link: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data

3. Chest X-Ray Images (Pneumonia) (Image, Classification)
•	Scenario: A radiologist is using an AI model as a "second opinion" to detect pneumonia. To trust the AI's diagnosis (a "black box" prediction from pixels), the radiologist needs to see what in the image led to the prediction.

•	Problem (Business): The task is binary classification (Pneumonia vs. Normal). A model can be accurate, but it's unusable in medicine if it's not interpretable. The doctor must know why the model made its decision.

•	Problem (Dataset): This is a large image dataset (5,863 images). The main challenge is the preprocessing required for sklearn models. You cannot feed raw images to an MLPClassifier. You must:

*	Resize all images to a uniform, small dimension (e.g., 64x64).

*	Convert images to grayscale.

*	Flatten each (64, 64) image into a 1D vector of 4,096 features (pixels).
This flattened vector is what you will train your sklearn models on. The data is also imbalanced.

Solution (XAI Application):
Global: 
•	PDP, ICE, and Permutation Importance are not useful here. A plot of the "average effect of pixel #2048" is meaningless.
Local (for the radiologist):
*	LIME (for Images): This is the main tool. LIME will not use the flattened vector. You will give it the original 2D image. It will highlight the "superpixels" (regions) in the lung that the model (which was trained on flattened data) used to make its "Pneumonia" diagnosis. This allows the radiologist to visually confirm if the model is focusing on a valid area of infection.
•	Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

