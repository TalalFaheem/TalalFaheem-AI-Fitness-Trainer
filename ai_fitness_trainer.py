import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import sqlite3
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create the database if it doesn't exist
def create_db():
    conn = sqlite3.connect('fitness_trainer.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    age INTEGER,
                    weight INTEGER,
                    height INTEGER,
                    fitness_level TEXT,
                    goal TEXT)''')
    conn.commit()
    conn.close()

# Function to authenticate users
def authenticate_user(username, password):
    conn = sqlite3.connect('fitness_trainer.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Function to save or update user data
def save_user_data(username, age, weight, height, fitness_level, goal):
    conn = sqlite3.connect('fitness_trainer.db')
    c = conn.cursor()

    # Check if the username already exists
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    existing_user = c.fetchone()

    if existing_user:
        # If user exists, update the data
        c.execute('''UPDATE users
                     SET age=?, weight=?, height=?, fitness_level=?, goal=?
                     WHERE username=?''',
                  (age, weight, height, fitness_level, goal, username))
    else:
        # If user does not exist, insert the new data
        c.execute('''INSERT INTO users (username, age, weight, height, fitness_level, goal)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (username, age, weight, height, fitness_level, goal))

    conn.commit()
    conn.close()

# Diet Recommendations based on fitness goals
def get_diet_recommendation(goal):
    if goal == "Lose Weight":
        return "High Protein, Low Carb, Calorie Deficit"
    elif goal == "Build Muscle":
        return "High Protein, Moderate Carbs, Calorie Surplus"
    elif goal == "Stay Fit":
        return "Balanced Diet with adequate Protein, Carbs, and Fats"
    else:
        return "General healthy diet"

# Advanced Workout Routines stored in a dictionary
workout_routines = {
    "Lose Weight": ["Cardio (30 min)", "HIIT (20 min)", "Strength Training (Full Body)"],
    "Build Muscle": ["Weight Lifting (3x sets)", "Strength Training (Split Focus)", "HIIT (15 min)"],
    "Stay Fit": ["Yoga (30 min)", "Running (30 min)", "Cycling (30 min)"]
}

# Generate random data for simulation
def generate_random_data():
    age = random.randint(18, 60)
    weight = random.randint(50, 120)
    height = random.randint(150, 190)
    activity_level = random.randint(0, 2)
    fitness_goal = random.choice(["Lose Weight", "Build Muscle", "Stay Fit"])
    return age, weight, height, activity_level, fitness_goal

# ML model setup (KNN and Naive Bayes)
def train_models():
    # Random Data for training models (simulating real-world data)
    data = {
        'Age': [22, 25, 30, 35, 40, 45, 50, 55, 60],
        'Weight': [60, 70, 80, 90, 100, 110, 120, 85, 75],
        'Height': [160, 170, 180, 175, 165, 178, 185, 160, 172],
        'Activity Level': [0, 1, 2, 1, 0, 2, 1, 1, 2],
        'Goal': ["Lose Weight", "Build Muscle", "Stay Fit", "Lose Weight", "Build Muscle", "Stay Fit", "Lose Weight", "Stay Fit", "Build Muscle"]
    }
    df = pd.DataFrame(data)
    X = df[['Age', 'Weight', 'Height', 'Activity Level']]
    y = df['Goal']

    # Standardize the data (ensure it is a DataFrame)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the models
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_scaled, y)

    nb_model = GaussianNB()
    nb_model.fit(X_scaled, y)

    return knn_model, nb_model, scaler

# Load the models
knn_model, nb_model, scaler = train_models()

# Function to predict using the machine learning models
def predict_fitness_level(age, weight, height, activity_level):
    input_data = pd.DataFrame([[age, weight, height, activity_level]], columns=['Age', 'Weight', 'Height', 'Activity Level'])
    input_data_scaled = scaler.transform(input_data)

    # Predict with both models
    knn_prediction = knn_model.predict(input_data_scaled)[0]
    nb_prediction = nb_model.predict(input_data_scaled)[0]

    return knn_prediction, nb_prediction

# Genetic Algorithm for workout optimization
def genetic_algorithm(goal):
    exercises = workout_routines[goal]
    population = [[random.choice(exercises) for _ in range(3)] for _ in range(10)]

    def fitness(plan):
        return len(set(plan))  # Maximize diversity in the workout

    for _ in range(10):  # Evolution over 10 generations
        population = sorted(population, key=fitness, reverse=True)
        next_gen = population[:5]  # Select top 5

        while len(next_gen) < 10:
            parent1, parent2 = random.sample(population[:5], 2)
            crossover_point = random.randint(1, 2)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            if random.random() < 0.1:  # Mutation
                child[random.randint(0, 2)] = random.choice(exercises)
            next_gen.append(child)

        population = next_gen

    return population[0]  # Return the best plan

# GUI setup
root = tk.Tk()
root.title("AI-Powered Personalized Fitness Trainer")
root.geometry("800x600")

# Create the database
create_db()

# Global variable for username
username = ""

# User authentication window
def show_login_window():
    global username
    login_window = tk.Toplevel(root)
    login_window.title("Login")
    login_window.geometry("300x200")

    tk.Label(login_window, text="Username").pack()
    username_entry = tk.Entry(login_window)
    username_entry.pack()

    tk.Label(login_window, text="Password").pack()
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack()

    def login_action():
        global username
        user = authenticate_user(username_entry.get(), password_entry.get())
        if user:
            username = username_entry.get()
            login_window.destroy()
            show_main_window()
        else:
            messagebox.showerror("Login Failed", "Invalid credentials")

    tk.Button(login_window, text="Login", command=login_action).pack()

# Main window after login
def show_main_window():
    def get_recommendations():
        # Get data entered by the user
        age = int(age_entry.get())
        weight = int(weight_entry.get())
        height = int(height_entry.get())
        activity_level = int(activity_level_var.get())
        goal = goal_var.get()

        # Use the models to predict the fitness level
        knn_prediction, nb_prediction = predict_fitness_level(age, weight, height, activity_level)

        # Optimize workout routine using Genetic Algorithm
        optimized_workout = genetic_algorithm(goal)

        # Save or update user data
        save_user_data(username, age, weight, height, knn_prediction, goal)

        # Display results
        messagebox.showinfo("Recommendations",
                            f"KNN Prediction: {knn_prediction}\nNaive Bayes Prediction: {nb_prediction}\nOptimized Workout: {', '.join(optimized_workout)}\nDiet: {get_diet_recommendation(goal)}")

    main_window = tk.Toplevel(root)
    main_window.title("Fitness Plan")
    main_window.geometry("600x400")

    tk.Label(main_window, text="Enter Your Data").pack()

    # Input fields for data
    tk.Label(main_window, text="Age").pack()
    age_entry = tk.Entry(main_window)
    age_entry.pack()

    tk.Label(main_window, text="Weight (kg)").pack()
    weight_entry = tk.Entry(main_window)
    weight_entry.pack()

    tk.Label(main_window, text="Height (cm)").pack()
    height_entry = tk.Entry(main_window)
    height_entry.pack()

    tk.Label
