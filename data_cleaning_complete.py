#!/usr/bin/env python3
"""
Data Cleaning & Integration for MovieLens 100K SVD Model
เป้าหมาย: รวม u.data + u.item + u.user → ได้ final_data ที่สะอาดและพร้อมใช้งาน
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime

def setup_paths():
    """Set up project and data paths"""
    project_root = os.path.abspath(".") if not os.getcwd().endswith("notebooks") else os.path.abspath("..")
    data_dir = os.path.join(project_root, "data")
    print("PROJECT_ROOT:", project_root)
    print("DATA_DIR:", data_dir)
    return project_root, data_dir

def load_raw_data(data_dir):
    """Load raw MovieLens 100K data files"""
    print("=== Loading Raw Data ===")
    
    # Load ratings data
    ratings = pd.read_csv(
        os.path.join(data_dir, "u.data"),
        sep="\t", 
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    # Load items data with proper encoding
    items = pd.read_csv(
        os.path.join(data_dir, "u.item"),
        sep="|", 
        encoding="latin-1",
        names=[
            "item_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ],
        na_values=["", " ", "None"]
    )
    
    # Clean and parse release dates
    s = items["release_date"].astype(str).str.strip()
    s = s.str.replace(r"[^0-9A-Za-z\-]", "", regex=True)
    items["release_date"] = pd.to_datetime(s, errors="coerce", dayfirst=True)
    
    # Load users data
    users = pd.read_csv(
        os.path.join(data_dir, "u.user"),
        sep="|", 
        names=["user_id", "age", "gender", "occupation", "zip_code"]
    )
    
    print(f"Loaded - Ratings: {ratings.shape}, Items: {items.shape}, Users: {users.shape}")
    return ratings, items, users

def inspect_data_quality(ratings, items, users):
    """Inspect data quality and report issues"""
    print("\n=== Data Quality Inspection ===")
    
    # Check sizes
    print(f"Ratings shape: {ratings.shape}")
    print(f"Items shape: {items.shape}")
    print(f"Users shape: {users.shape}")
    
    # Check missing values
    print("\n=== Missing Values ===")
    print(f"Ratings missing: {ratings.isnull().sum().sum()}")
    print(f"Items missing: {items.isnull().sum().sum()}")
    print(f"Users missing: {users.isnull().sum().sum()}")
    
    # Check unique values
    print("\n=== Unique Counts ===")
    print(f"Unique users: {ratings['user_id'].nunique()}")
    print(f"Unique items: {ratings['item_id'].nunique()}")
    print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    
    # Check duplicates
    print("\n=== Duplicate Check ===")
    print(f"Duplicate ratings: {ratings.duplicated().sum()}")
    print(f"Duplicate items: {items.duplicated().sum()}")
    print(f"Duplicate users: {users.duplicated().sum()}")

def clean_ratings_data(ratings):
    """Clean ratings data"""
    print("\n=== Cleaning Ratings Data ===")
    
    # Remove duplicates
    before_ratings = len(ratings)
    ratings = ratings.drop_duplicates()
    print(f"Removed {before_ratings - len(ratings)} duplicate ratings")
    
    # Check for invalid ratings
    invalid_ratings = ratings[(ratings['rating'] < 1) | (ratings['rating'] > 5)]
    print(f"Invalid ratings found: {len(invalid_ratings)}")
    if len(invalid_ratings) > 0:
        print(invalid_ratings)
        ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]
    
    # Convert timestamp to datetime
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    print(f"Final ratings shape: {ratings.shape}")
    return ratings

def clean_items_data(items):
    """Clean items data"""
    print("\n=== Cleaning Items Data ===")
    
    # Remove duplicates
    before_items = len(items)
    items = items.drop_duplicates(subset=['item_id'])
    print(f"Removed {before_items - len(items)} duplicate items")
    
    # Clean movie titles
    items['movie_title'] = items['movie_title'].str.strip()
    
    # Extract year from movie title
    items['release_year'] = items['movie_title'].str.extract(r'\((\d{4})\)$')
    items['release_year'] = pd.to_numeric(items['release_year'], errors='coerce')
    
    # Clean genre columns
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    for col in genre_cols:
        items[col] = items[col].astype(int)
    
    # Create genre list for each movie
    def get_genres(row):
        genres = []
        for genre in genre_cols[1:]:  # Skip 'unknown'
            if row[genre] == 1:
                genres.append(genre)
        return ', '.join(genres) if genres else 'Unknown'
    
    items['genres'] = items.apply(get_genres, axis=1)
    
    print(f"Final items shape: {items.shape}")
    return items

def clean_users_data(users):
    """Clean users data"""
    print("\n=== Cleaning Users Data ===")
    
    # Remove duplicates
    before_users = len(users)
    users = users.drop_duplicates(subset=['user_id'])
    print(f"Removed {before_users - len(users)} duplicate users")
    
    # Clean occupation
    users['occupation'] = users['occupation'].str.strip().str.lower()
    
    # Clean gender
    users['gender'] = users['gender'].str.upper()
    
    # Create age groups
    def categorize_age(age):
        if age < 18:
            return 'Under 18'
        elif age < 25:
            return '18-24'
        elif age < 35:
            return '25-34'
        elif age < 50:
            return '35-49'
        elif age < 65:
            return '50-64'
        else:
            return '65+'
    
    users['age_group'] = users['age'].apply(categorize_age)
    
    print(f"Final users shape: {users.shape}")
    return users

def integrate_data(ratings, items, users):
    """Integrate all data sources"""
    print("\n=== Data Integration ===")
    
    # Start with ratings as base
    final_data = ratings.copy()
    
    # Join with items data
    final_data = final_data.merge(
        items[['item_id', 'movie_title', 'release_year', 'genres']], 
        on='item_id', 
        how='left'
    )
    
    # Join with users data
    final_data = final_data.merge(
        users[['user_id', 'age', 'gender', 'occupation', 'age_group']], 
        on='user_id', 
        how='left'
    )
    
    print(f"Final integrated data shape: {final_data.shape}")
    print(f"Columns: {list(final_data.columns)}")
    
    # Check missing values after integration
    print("\n=== Missing Values After Integration ===")
    print(final_data.isnull().sum())
    
    return final_data

def prepare_svd_data(final_data):
    """Prepare data specifically for SVD model training"""
    print("\n=== SVD Data Preparation ===")
    
    # 1. Create user-item matrix
    user_item_matrix = final_data.pivot_table(
        index='user_id', 
        columns='item_id', 
        values='rating'
    ).fillna(0)
    
    print(f"User-Item Matrix shape: {user_item_matrix.shape}")
    
    # 2. Create SVD-ready data
    svd_data = final_data[['user_id', 'item_id', 'rating']].copy()
    
    # 3. Data statistics for SVD
    print("\n=== SVD Data Statistics ===")
    print(f"Total ratings: {len(svd_data):,}")
    print(f"Unique users: {svd_data['user_id'].nunique():,}")
    print(f"Unique items: {svd_data['item_id'].nunique():,}")
    print(f"Rating distribution:")
    print(svd_data['rating'].value_counts().sort_index())
    
    # 4. Data sparsity calculation
    total_possible_ratings = svd_data['user_id'].nunique() * svd_data['item_id'].nunique()
    actual_ratings = len(svd_data)
    sparsity = (1 - actual_ratings / total_possible_ratings) * 100
    print(f"\nData sparsity: {sparsity:.2f}%")
    
    return user_item_matrix, svd_data

def save_cleaned_data(data_dir, final_data, svd_data, user_item_matrix):
    """Save all cleaned data"""
    print("\n=== Saving Cleaned Data ===")
    
    # Save final integrated data
    final_data_path = os.path.join(data_dir, 'final_data_cleaned.csv')
    final_data.to_csv(final_data_path, index=False)
    print(f"Final data saved to: {final_data_path}")
    
    # Save SVD-ready data
    svd_data_path = os.path.join(data_dir, 'svd_data.csv')
    svd_data.to_csv(svd_data_path, index=False)
    print(f"SVD data saved to: {svd_data_path}")
    
    # Save user-item matrix
    matrix_path = os.path.join(data_dir, 'user_item_matrix.csv')
    user_item_matrix.to_csv(matrix_path)
    print(f"User-Item matrix saved to: {matrix_path}")
    
    print("\n✅ Data cleaning and preparation completed!")
    print("Ready for SVD model training in the next notebook.")

def main():
    """Main data cleaning pipeline"""
    print("🚀 Starting MovieLens 100K Data Cleaning & Integration")
    print("=" * 60)
    
    # 1. Setup paths
    project_root, data_dir = setup_paths()
    
    # 2. Load raw data
    ratings, items, users = load_raw_data(data_dir)
    
    # 3. Inspect data quality
    inspect_data_quality(ratings, items, users)
    
    # 4. Clean individual datasets
    ratings_clean = clean_ratings_data(ratings)
    items_clean = clean_items_data(items)
    users_clean = clean_users_data(users)
    
    # 5. Integrate data
    final_data = integrate_data(ratings_clean, items_clean, users_clean)
    
    # 6. Prepare SVD data
    user_item_matrix, svd_data = prepare_svd_data(final_data)
    
    # 7. Save cleaned data
    save_cleaned_data(data_dir, final_data, svd_data, user_item_matrix)
    
    # 8. Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Total ratings processed: {len(final_data):,}")
    print(f"✅ Unique users: {final_data['user_id'].nunique():,}")
    print(f"✅ Unique movies: {final_data['item_id'].nunique():,}")
    print(f"✅ Data ready for SVD training!")
    
    return final_data, svd_data, user_item_matrix

if __name__ == "__main__":
    final_data, svd_data, user_item_matrix = main()