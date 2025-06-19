"""
api_client.py

TMDB API client using v3 API key (query parameter-based auth).
"""

import os
import requests
import time
import logging
import json
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise EnvironmentError("TMDB_API_KEY is not set in the environment or .env file.")

BASE_URL = "https://api.themoviedb.org/3"
RETRIES = 3
WAIT_TIME = 1

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _handle_request(endpoint, params=None):
    if params is None:
        params = {}
    params["api_key"] = TMDB_API_KEY
    url = f"{BASE_URL}{endpoint}"

    for attempt in range(RETRIES):
        try:
            response = requests.get(url, params=params)
            logger.debug(f"Request URL: {response.url}")
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                logger.warning(f"Rate limited. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
            elif response.status_code == 200:
                return response.json()
            else:
                logger.error(f"TMDB API error {response.status_code}: {response.text}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Request exception: {e}, retrying...")
            time.sleep(WAIT_TIME)
    logger.error("Exceeded max retries.")
    return None


@lru_cache(maxsize=500)
def search_movie(title):
    """Search TMDB for a movie by title."""
    result = _handle_request("/search/movie", {"query": title})
    return result.get("results", []) if result else []


@lru_cache(maxsize=500)
def get_movie_details(tmdb_id):
    """Fetch full movie metadata from TMDB."""
    return _handle_request(f"/movie/{tmdb_id}")


@lru_cache(maxsize=500)
def get_recommendations(tmdb_id):
    """Fetch TMDB movie recommendations based on a movie ID."""
    result = _handle_request(f"/movie/{tmdb_id}/recommendations")
    return result.get("results", []) if result else []


def map_movielens_to_tmdb(movies_df):
    """Map MovieLens movie titles to TMDB IDs."""
    cache_file = "data/processed/movieId_tmdbId_map.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    mapping = {}
    for _, row in movies_df.iterrows():
        title = row["title"].split(" (")[0]
        movie_id = row["movieId"]
        results = search_movie(title)
        if results:
            mapping[movie_id] = results[0]["id"]
        time.sleep(0.25)

    with open(cache_file, "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping
