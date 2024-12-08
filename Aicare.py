import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import os.path  # Operating system interactions

import matplotlib.pyplot as plt # Plotting library

import seaborn as sns  # Statistical data visualization

from sklearn.preprocessing import MinMaxScaler  # Scaling features to a range
from sklearn.model_selection import train_test_split  # Splitting data into training and test sets
from sklearn.metrics.pairwise import cosine_similarity  # Calculating cosine similarity between vectors

import tensorflow as tf  # TensorFlow library for machine learning
from tensorflow.keras.models import Sequential  # Sequential model from Keras
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout  # Layers for the neural network
from tensorflow.keras.optimizers import Adam  # Optimizer for training the model
from tensorflow.keras import regularizers  # Import regularizers for the model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


class RecipeRecommender:
    def __init__(self, csv_path='./recipes.csv', model_path='./myModel.h5'):
        self.csv_path = csv_path
        self.model_path = model_path
        self.data = pd.read_csv(self.csv_path)
        self.df = self.data.copy()
        self.selected_columns = [
            'RecipeId', 'Calories', 'FatContent', 'SaturatedFatContent',
            'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
            'FiberContent', 'SugarContent', 'ProteinContent'
        ]
        self.filter_columns = [
            'Calories', 'FatContent', 'SaturatedFatContent',
            'CholesterolContent', 'SodiumContent',
            'CarbohydrateContent', 'FiberContent',
            'SugarContent', 'ProteinContent'
        ]
        self.features_to_compare = [
            'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent',
            'SugarContent', 'ProteinContent'
        ]
        self.df = self.df[self.selected_columns]
        self.scaler = MinMaxScaler()
        self.recommendations_dict = {}
        self.model = None

    def preprocess_data(self, sample_size=6000):
        percentile_95_thresholds = self.df[self.filter_columns].quantile(0.95)
        filtered_df = self.df[(self.df[self.filter_columns] <= percentile_95_thresholds).all(axis=1)]

        if sample_size > len(filtered_df):
            print("Warning: Sample size reduced to DataFrame size to avoid error.")
            sample_size = len(filtered_df)

        sampled_df = filtered_df.sample(sample_size)
        X = sampled_df.drop(columns=['RecipeId'])
        y = sampled_df['RecipeId']

        X_scaled = self.scaler.fit_transform(X)
        X_train_scaled, X_temp_scaled, y_train, y_temp = train_test_split(
            X_scaled, y, train_size=0.8, random_state=42
        )
        X_eval_scaled, X_test_scaled, y_eval, y_test = train_test_split(
            X_temp_scaled, y_temp, train_size=0.5, random_state=42
        )

        print("Training set shape:", X_train_scaled.shape)
        print("Development set shape:", X_eval_scaled.shape)
        print("Test set shape:", X_test_scaled.shape)

        return X_train_scaled, X_eval_scaled, X_test_scaled, y_train, y_eval, y_test

    def load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        self.model = load_model(self.model_path, custom_objects={'mse': MeanSquaredError()})
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError(), metrics=['mae'])

    def compute_similarity(self, X_train_scaled):
        predicted_latent_features = self.model.predict(X_train_scaled)
        similarity_matrix = np.dot(predicted_latent_features, predicted_latent_features.T)
        norms = np.linalg.norm(predicted_latent_features, axis=1, keepdims=True)
        similarity_matrix /= np.dot(norms, norms.T)
        return similarity_matrix

    def build_recommendations(self, similarity_matrix, X_train_scaled):
        for item_index in range(len(X_train_scaled)):
            sorted_indices = similarity_matrix[item_index].argsort()[::-1]
            similar_items_list = [(similarity_matrix[item_index][idx], idx) for idx in sorted_indices if idx != item_index]
            self.recommendations_dict[item_index] = similar_items_list

        items_to_display = 10
        for index, (item_id, similar_items) in enumerate(self.recommendations_dict.items()):
            if index < items_to_display:
                print(f"Item {item_id}: {similar_items}")
                print('*' * 100)
            else:
                break

    def compute_bmr(self,gender, body_weight, body_height, age):
        """
        Calculate Basal Metabolic Rate (BMR) based on gender, body weight, body height, and age.

        Args:
            gender (str): Gender of the individual ('male' or 'female').
            body_weight (float): Body weight of the individual in kilograms.
            body_height (float): Body height of the individual in centimeters.
            age (int): Age of the individual in years.

        Return:
            float: Basal Metabolic Rate (BMR) value.
        """
        if gender == 'male':
            # For Men: BMR = (10 x weight in kg) + (6.25 x height in cm) - (5 x age in years) + 5
            bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age + 5
        elif gender == 'female':
            # For Women: BMR = (10 x weight in kg) + (6.25 x height in cm) - (5 x age in years) - 161
            bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age - 161
        else:
            raise ValueError("Invalid gender. Please choose 'male' or 'female'.")
        return bmr_value

    def compute_daily_caloric_intake(self,bmr, activity_intensity, objective):
        """
        Calculate total daily caloric intake based on Basal Metabolic Rate (BMR), activity level, and personal goal.

        Args:
            bmr (float): Basal Metabolic Rate (BMR) value.
            activity_intensity (str): Activity level of the individual ('sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extra_active').
            objective (str): Personal goal of the individual ('weight_loss', 'muscle_gain', 'health_maintenance').

        Return:
            int: Total daily caloric intake.
        """
        # Define activity multipliers based on intensity
        intensity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extra_active': 1.9
        }

        # Define goal adjustments based on objective
        objective_adjustments = {
            'weight_loss': 0.8,
            'muscle_gain': 1.2,
            'health_maintenance': 1
        }

        # Calculate maintenance calories based on activity intensity
        maintenance_calories = bmr * intensity_multipliers[activity_intensity]

        # Adjust maintenance calories based on personal objective
        total_caloric_intake = maintenance_calories * objective_adjustments[objective]

        return round(total_caloric_intake)

    def suggest_recipes(self,category, body_weight, body_height, age, activity_intensity, objective):
        """
        Generate food recommendations based on the user's profile and dietary goals.

        Args:
            category (str): Gender category of the user ('male' or 'female').
            body_weight (float): Weight of the user in kilograms.
            body_height (float): Height of the user in centimeters.
            age (int): Age of the user in years.
            activity_intensity (str): Physical activity level of the user ('sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extra_active').
            objective (str): Dietary objective of the user ('weight_loss', 'muscle_gain', 'health_maintenance').

        Return:
            pd.DataFrame: Recommended recipes including name and calorie content.
        """
        X_train_scaled, X_eval_scaled, X_test_scaled, y_train, y_eval, y_test = self.preprocess_data()
        self.load_model()
        similarity_matrix = self.compute_similarity(X_train_scaled)
        self.build_recommendations(similarity_matrix, X_train_scaled)
        # Calculate the Basal Metabolic Rate (BMR) for the user
        bmr = self.compute_bmr(category, body_weight, body_height, age)

        # Calculate the total daily caloric intake based on activity intensity and dietary objective
        total_calories = self.compute_daily_caloric_intake(bmr, activity_intensity, objective)

        # Prepare input data for the model with desired total calories
        user_input_features = np.array([[total_calories, 0, 0, 0, 0, 0, 0, 0, 0]])

        # Scale the input data to match the model's training scale
        scaled_input_features = self.scaler.transform(user_input_features)

        # Predict latent features for the input data
        predicted_latent_features = self.model.predict(scaled_input_features)

        # Find the index with the highest prediction probability
        top_prediction_index = np.argmax(predicted_latent_features.flatten())

        # Retrieve recommended recipes based on the highest prediction
        similar_recipe_indices = np.array(self.recommendations_dict[top_prediction_index])
        # recommended_recipes = data.iloc[similar_recipe_indices[:, 1].astype(int)][['Name', 'Calories']]
        recommended_recipes = self.data.iloc[y_train.index[similar_recipe_indices[:, 1].astype(int)]][['Name', 'Calories']]

        return recommended_recipes.head(10)  # Return the top 5 recommended recipes


# if __name__ == "__main__":
#     recomen = RecipeRecommender()

#     user_category = input("Sex (male/female): ")
#     user_body_weight = int(input("Weight (kg): "))
#     user_body_height = int(input("Height (cm): "))
#     user_age = int(input("Age (in year): "))
#     user_activity_intensity = input("Activity (sedentary/lightly_active/moderately_active/very_active/extra_active): ")
#     user_objective = input("Diet Objective (weight_loss/muscle_gain/health_maintenance): ")

#     # Generate suggested recipes
#     suggested_recipes = recomen.suggest_recipes(
#         category=user_category,
#         body_weight=user_body_weight,
#         body_height=user_body_height,
#         age=user_age,
#         activity_intensity=user_activity_intensity,
#         objective=user_objective
#     )

#     # Calculate required daily calories
#     required_calories = recomen.compute_daily_caloric_intake(
#         bmr=recomen.compute_bmr(user_category, user_body_weight, user_body_height, user_age),
#         activity_intensity=user_activity_intensity,
#         objective=user_objective
#     )

#     # Print the required calories
#     print(f"Required Daily Calories: {required_calories}\n")
#     print("Top 10 Suggested Recipes:\n")

#     # Print the suggested recipes in a readable format
#     for idx, recipe in suggested_recipes.iterrows():
#         print(f"{idx + 1}. {recipe['Name']} - {recipe['Calories']} Calories")
#         print('-' * 40)