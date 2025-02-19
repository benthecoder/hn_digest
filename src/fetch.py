import requests
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HNFetcher:
    """Simple HN fetcher for daily top stories"""

    def __init__(self):
        self.base_url = "https://hacker-news.firebaseio.com/v0"

    def _get_item(self, item_id: int) -> dict:
        """Fetch a single item from HN API"""
        url = f"{self.base_url}/item/{item_id}.json"
        try:
            response = requests.get(url, timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Error fetching item {item_id}: {str(e)}")
            return None

    def get_daily_top(self, limit: int = 100) -> pd.DataFrame:
        """Fetch top stories of the day"""
        # Get top story IDs
        response = requests.get(f"{self.base_url}/topstories.json")
        if response.status_code != 200:
            raise Exception("Failed to fetch top stories")

        story_ids = response.json()[:limit]
        stories = []

        for story_id in story_ids:
            story = self._get_item(story_id)
            if story and story.get("type") == "story":
                stories.append(
                    {
                        "id": story.get("id"),
                        "title": story.get("title"),
                        "url": story.get("url", ""),
                        "score": story.get("score", 0),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }
                )

        return pd.DataFrame(stories)


if __name__ == "__main__":
    fetcher = HNFetcher()
    df = fetcher.get_daily_top()

    # Print stats
    print(f"\nFetched {len(df)} stories on {datetime.now().strftime('%Y-%m-%d')}")

    # Save to CSV with date in filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(f"data/hn_top_{date_str}.csv", index=False)

    # Display the stories
    print("\nToday's Top Stories:")
    for idx, row in df.iterrows():
        print(f"{idx + 1}. {row['title']} (Score: {row['score']})")
