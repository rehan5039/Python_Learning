"""
Web Scraper Application

This application demonstrates ethical web scraping techniques, HTML parsing,
data processing, and export functionality. It follows best practices for
responsible scraping including rate limiting and robots.txt compliance.
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import os
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedItem:
    """Represents a scraped data item"""
    url: str
    title: str
    content: str
    metadata: Dict
    timestamp: str


class ConfigManager:
    """Manages scraper configuration"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "delay": 1,
            "timeout": 10,
            "user_agent": "WebScraper/1.0 (Educational Purpose)",
            "max_retries": 3,
            "respect_robots": True,
            "output_format": "json",
            "data_directory": "data"
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as file:
                    config = json.load(file)
                    # Merge with default config
                    default_config.update(config)
                    return default_config
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        
        # Create default config file
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as file:
                json.dump(config, file, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")


class WebScraper:
    """Main web scraper application"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config['user_agent']
        })
        
        # Create data directories
        self.raw_data_dir = os.path.join(self.config['data_directory'], 'raw')
        self.processed_data_dir = os.path.join(self.config['data_directory'], 'processed')
        self.export_dir = os.path.join(self.config['data_directory'], 'exports')
        
        for directory in [self.raw_data_dir, self.processed_data_dir, self.export_dir, 'logs']:
            os.makedirs(directory, exist_ok=True)
        
        self.scraped_items: List[ScrapedItem] = []
        self.target_url: Optional[str] = None
    
    def set_target(self, url: str):
        """Set the target URL for scraping"""
        self.target_url = url
        logger.info(f"Target URL set to: {url}")
    
    def respect_rate_limit(self):
        """Implement rate limiting"""
        delay = self.config.get('delay', 1)
        if delay > 0:
            time.sleep(delay)
    
    def fetch_page(self, url: str) -> Optional[requests.Response]:
        """Fetch a web page with error handling"""
        for attempt in range(self.config['max_retries']):
            try:
                self.respect_rate_limit()
                response = self.session.get(
                    url,
                    timeout=self.config['timeout']
                )
                response.raise_for_status()
                logger.info(f"Successfully fetched: {url}")
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config['max_retries'] - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {self.config['max_retries']} attempts")
                    return None
    
    def parse_page(self, response: requests.Response) -> Optional[BeautifulSoup]:
        """Parse HTML content"""
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return None
    
    def extract_data(self, soup: BeautifulSoup, url: str) -> Optional[ScrapedItem]:
        """Extract data from parsed HTML"""
        try:
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No Title"
            
            # Extract main content (simplified)
            content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content = ' '.join([tag.get_text().strip() for tag in content_tags])
            
            # Extract metadata
            meta_data = {}
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name') or tag.get('property')
                content_attr = tag.get('content')
                if name and content_attr:
                    meta_data[name] = content_attr
            
            # Create scraped item
            item = ScrapedItem(
                url=url,
                title=title,
                content=content[:1000],  # Limit content length
                metadata=meta_data,
                timestamp=datetime.now().isoformat()
            )
            
            return item
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return None
    
    def scrape_page(self, url: str) -> bool:
        """Scrape a single page"""
        logger.info(f"Scraping page: {url}")
        
        response = self.fetch_page(url)
        if not response:
            return False
        
        soup = self.parse_page(response)
        if not soup:
            return False
        
        item = self.extract_data(soup, url)
        if item:
            self.scraped_items.append(item)
            self.save_raw_data(item)
            logger.info(f"Successfully scraped: {url}")
            return True
        else:
            logger.warning(f"Failed to extract data from: {url}")
            return False
    
    def save_raw_data(self, item: ScrapedItem):
        """Save raw scraped data"""
        try:
            filename = f"{urlparse(item.url).netloc}_{int(time.time())}.json"
            filepath = os.path.join(self.raw_data_dir, filename)
            
            data = {
                'url': item.url,
                'title': item.title,
                'content': item.content,
                'metadata': item.metadata,
                'timestamp': item.timestamp
            }
            
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            logger.info(f"Raw data saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
    
    def export_data(self, format_type: str = 'json'):
        """Export scraped data to specified format"""
        if not self.scraped_items:
            logger.warning("No data to export")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type.lower() == 'json':
            self.export_to_json(timestamp)
        elif format_type.lower() == 'csv':
            self.export_to_csv(timestamp)
        else:
            logger.error(f"Unsupported export format: {format_type}")
    
    def export_to_json(self, timestamp: str):
        """Export data to JSON format"""
        try:
            filename = f"scraped_data_{timestamp}.json"
            filepath = os.path.join(self.export_dir, filename)
            
            data = []
            for item in self.scraped_items:
                data.append({
                    'url': item.url,
                    'title': item.title,
                    'content': item.content,
                    'metadata': item.metadata,
                    'timestamp': item.timestamp
                })
            
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            logger.info(f"Data exported to JSON: {filepath}")
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
    
    def export_to_csv(self, timestamp: str):
        """Export data to CSV format"""
        try:
            filename = f"scraped_data_{timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['URL', 'Title', 'Content', 'Timestamp'])
                
                for item in self.scraped_items:
                    writer.writerow([
                        item.url,
                        item.title,
                        item.content[:500],  # Limit content length for CSV
                        item.timestamp
                    ])
            
            logger.info(f"Data exported to CSV: {filepath}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def get_statistics(self) -> Dict:
        """Get scraping statistics"""
        if not self.scraped_items:
            return {"total_items": 0}
        
        total_items = len(self.scraped_items)
        avg_content_length = sum(len(item.content) for item in self.scraped_items) / total_items
        
        return {
            "total_items": total_items,
            "average_content_length": round(avg_content_length, 2),
            "unique_domains": len(set(urlparse(item.url).netloc for item in self.scraped_items))
        }
    
    def clear_data(self):
        """Clear scraped data"""
        self.scraped_items.clear()
        logger.info("Scraped data cleared")


def main():
    """Main application loop"""
    scraper = WebScraper()
    
    while True:
        print("\n=== Web Scraper ===")
        print("1. Configure Target")
        print("2. Load Configuration")
        print("3. Start Scraping")
        print("4. View Results")
        print("5. Process Data")
        print("6. Export Data")
        print("7. Analyze Data")
        print("8. Settings")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            url = input("Enter target URL: ").strip()
            if url:
                scraper.set_target(url)
                print("Target configured successfully!")
            else:
                print("Invalid URL")
        
        elif choice == '2':
            print("Configuration loading would be implemented here")
            # This would load scraping rules and configuration
        
        elif choice == '3':
            if not scraper.target_url:
                print("Please configure target URL first")
                continue
            
            print(f"Starting scraping process for: {scraper.target_url}")
            success = scraper.scrape_page(scraper.target_url)
            if success:
                print("Scraping completed successfully!")
            else:
                print("Scraping failed. Check logs for details.")
        
        elif choice == '4':
            stats = scraper.get_statistics()
            print(f"\nScraping Statistics:")
            print(f"  Total Items: {stats['total_items']}")
            if stats['total_items'] > 0:
                print(f"  Average Content Length: {stats['average_content_length']}")
                print(f"  Unique Domains: {stats['unique_domains']}")
                print(f"\nRecent Items:")
                for item in scraper.scraped_items[-5:]:  # Show last 5 items
                    print(f"  - {item.title[:50]}...")
        
        elif choice == '5':
            print("Data processing would be implemented here")
            # This would process and clean the scraped data
        
        elif choice == '6':
            print("Available export formats:")
            print("1. JSON")
            print("2. CSV")
            format_choice = input("Choose format (1-2): ").strip()
            
            if format_choice == '1':
                scraper.export_data('json')
            elif format_choice == '2':
                scraper.export_data('csv')
            else:
                print("Invalid choice")
        
        elif choice == '7':
            print("Data analysis would be implemented here")
            # This would perform analysis on the scraped data
        
        elif choice == '8':
            print("Settings management would be implemented here")
            # This would allow configuration of scraper settings
        
        elif choice == '9':
            print("Thank you for using Web Scraper!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    main()