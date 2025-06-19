"""
data_loader.py

This module handles:
1. Loading MovieLens 100k dataset files
2. Initial data exploration
3. Data cleaning (filtering low-activity users/movies)
4. Saving cleaned datasets to 'data/processed'
5. Creating visualizations in 'notebooks/' for analysis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATA_PATH = "data/raw/ml-latest-small"
PROCESSED_DATA_PATH = "data/processed"
NOTEBOOKS_PATH = "notebooks"

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(NOTEBOOKS_PATH, exist_ok=True)

def load_datasets():
    """Load ratings, movies, and tags into DataFrames."""
    ratings = pd.read_csv(os.path.join(RAW_DATA_PATH, "ratings.csv"))
    movies = pd.read_csv(os.path.join(RAW_DATA_PATH, "movies.csv"))
    tags = pd.read_csv(os.path.join(RAW_DATA_PATH, "tags.csv"))
    return ratings, movies, tags

def explore_data(ratings, movies, tags):
    """Perform initial exploration of datasets."""
    print("Dataset Shapes:")
    print(f"Ratings: {ratings.shape}")
    print(f"Movies: {movies.shape}")
    print(f"Tags: {tags.shape}\n")

    print("Ratings Summary:")
    print(ratings.describe(), "\n")

    print("Missing Values:")
    print(ratings.isnull().sum())
    print(movies.isnull().sum())
    print(tags.isnull().sum(), "\n")

    print("Duplicate Entries:")
    print(f"Ratings: {ratings.duplicated().sum()}")
    print(f"Movies: {movies.duplicated().sum()}")
    print(f"Tags: {tags.duplicated().sum()}\n")

    print("Top 10 Most Rated Movies:")
    movie_counts = ratings['movieId'].value_counts().head(10)
    print(movie_counts)

    print("\nTop 10 Most Active Users:")
    user_counts = ratings['userId'].value_counts().head(10)
    print(user_counts)

def create_visualizations(ratings):
    """Generate and save plots related to the ratings."""
    plt.figure(figsize=(8, 5))
    sns.histplot(ratings['rating'], bins=10)
    plt.title("Rating Distribution")
    plt.savefig(f"{NOTEBOOKS_PATH}/rating_distribution.png")
    plt.close()

    movie_counts = ratings['movieId'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.histplot(movie_counts, bins=50, log_scale=(True, True))
    plt.title("Ratings per Movie (Log Scale)")
    plt.savefig(f"{NOTEBOOKS_PATH}/ratings_per_movie.png")
    plt.close()

    user_counts = ratings['userId'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.histplot(user_counts, bins=50, log_scale=(True, True))
    plt.title("Ratings per User (Log Scale)")
    plt.savefig(f"{NOTEBOOKS_PATH}/ratings_per_user.png")
    plt.close()

    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="count", y="mean", data=stats, alpha=0.6)
    plt.title("Average Rating vs Number of Ratings")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Average Rating")
    plt.savefig(f"{NOTEBOOKS_PATH}/avg_rating_vs_count.png")
    plt.close()

def clean_data(ratings, min_ratings=10, min_users=10):
    """Remove movies and users with too few ratings."""
    movie_counts = ratings['movieId'].value_counts()
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_ratings].index)]

    user_counts = ratings['userId'].value_counts()
    ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_users].index)]

    return ratings

def save_cleaned_data(ratings, movies, tags):
    """Save the cleaned datasets to processed directory."""
    ratings.to_csv(os.path.join(PROCESSED_DATA_PATH, "ratings_cleaned.csv"), index=False)
    movies.to_csv(os.path.join(PROCESSED_DATA_PATH, "movies.csv"), index=False)
    tags.to_csv(os.path.join(PROCESSED_DATA_PATH, "tags.csv"), index=False)

def run():
    """Run the full data loading, exploration, cleaning, and saving pipeline."""
    print("Loading datasets...")
    ratings, movies, tags = load_datasets()

    print("\nExploring data...")
    explore_data(ratings, movies, tags)

    print("\nCreating visualizations...")
    create_visualizations(ratings)

    print("\nCleaning data...")
    ratings_clean = clean_data(ratings)

    print("Saving cleaned data...")
    save_cleaned_data(ratings_clean, movies, tags)

    print("Data preparation complete.")

if __name__ == "__main__":
    run()
