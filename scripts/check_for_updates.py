"""Check USCIS for recent updates without full scraping."""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def check_recent_updates(days=30):
    """Check USCIS news and policy alerts for recent updates."""
    print("=" * 80)
    print("USCIS UPDATE CHECKER")
    print(f"Checking for updates in the last {days} days...")
    print("=" * 80)
    
    updates = []
    
    # Check news releases
    try:
        news_url = "https://www.uscis.gov/newsroom/news-releases"
        response = requests.get(news_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find recent news (adjust selectors based on actual page structure)
        news_items = soup.find_all('article', limit=10)
        
        print(f"\n📰 RECENT NEWS RELEASES:")
        for item in news_items[:5]:
            title = item.find('h2') or item.find('h3')
            if title:
                print(f"  • {title.get_text(strip=True)}")
                updates.append(("news", title.get_text(strip=True)))
    except Exception as e:
        print(f"  ⚠️  Could not fetch news: {e}")
    
    # Check policy alerts
    try:
        policy_url = "https://www.uscis.gov/policy-manual/policy-alerts"
        response = requests.get(policy_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        print(f"\n📋 RECENT POLICY ALERTS:")
        # Adjust selectors based on actual page
        alerts = soup.find_all('article', limit=10)
        
        for alert in alerts[:5]:
            title = alert.find('h2') or alert.find('h3')
            if title:
                print(f"  • {title.get_text(strip=True)}")
                updates.append(("policy", title.get_text(strip=True)))
    except Exception as e:
        print(f"  ⚠️  Could not fetch policy alerts: {e}")
    
    print("\n" + "=" * 80)
    
    if updates:
        print(f"\n✅ Found {len(updates)} recent updates")
        print("💡 Consider re-running the scraper to capture these changes")
    else:
        print("\n✅ No major updates detected")
    
    print("=" * 80)
    
    return updates

if __name__ == "__main__":
    check_recent_updates(30)

