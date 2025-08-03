# Import necessary libraries
import os
import json
from datetime import datetime
from urllib.parse import urlparse
from scrapy import Spider
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from app.services.indexer import add_document
from config import settings


class DataPipelines:
    """
    A Data Pipeline to process crawled items and save them to a file
    """

    def open_spider(self, spider):
        """
        Method called when the spider is opened
        """

        self.crawled_urls = []
        self.text_contents = []

    def process_item(self, item, spider):
        """
        Method to process each item scraped by the spider
        """

        self.crawled_urls.append(item['url'])
        self.text_contents.append(item['text'])

        return item

    def close_spider(self, spider):
        """
        Method called when the spider is closed
        """

        domain_name = urlparse(self.crawled_urls[0]).netloc.replace('.', '_')
        file_name = f"{domain_name}.txt"

        # Data directory to save the data
        data_dir = settings.data_directory_path
        os.makedirs(data_dir, exist_ok=True)

        # Save the file to the data directory
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as buffer:
            buffer.write("\n\n".join(self.text_contents))

        # Get the file size
        try:
            file_size = os.path.getsize(file_path)
        except Exception as e:
            file_size = 0

        # Save the file metadata to a JSON file
        json_path = file_path + ".json"
        metadata = {
            "file_name": domain_name,
            "file_extension": ".txt",
            "date": datetime.now().isoformat(),
            "size": file_size,
            "crawled_urls": self.crawled_urls
        }
        with open(json_path, 'w') as buffer:
            json.dump(metadata, buffer, indent=4)

        # Add the documents to the vector store
        add_document(file_path, domain_name, ".txt", 2500, 300)


class WebTextSpider(Spider):
    """
    A Scrapy spider to crawl a website and extract text content
    """

    # Name of the spider
    name = "web_text_spider"

    # Custom settings for the spider
    custom_settings={
        'LOG_LEVEL': 'INFO',
        'AUTOTHROTTLE_ENABLED': True,
        'DEPTH_LIMIT': 1,
        'ITEM_PIPELINES': {
            'app.services.crawler.DataPipelines': 1
        },
        'HTTPCACHE_ENABLED': False
    }

    # Link extractor to filter out unwanted urls
    link_extractor = LinkExtractor(
        deny_extensions=[
            'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
            'mp3', 'mp4', 'avi', 'mov'
        ],
        tags='a',
        attrs='href',
        unique=True
    )

    def __init__(self, *args, **kwargs):
        """
        Initialize the spider with allowed domains and start URLs
        """

        super(WebTextSpider, self).__init__(*args, **kwargs)

        self.allowed_domains = kwargs.get('allowed_domains', [])
        self.start_urls = kwargs.get('start_urls', [])

    def parse(self, response, **kwargs):
        """
        Parse the response and extract text content
        """

        self.logger.info(f"Crawling URL: {response.url}")

        # Check if the response URL is in the allowed domains
        if urlparse(response.url).netloc not in self.allowed_domains:
            self.logger.info(f"Skipping the not allowed URL: {response.url}")
            return

        # Check if the response is HTML content
        if not response.headers.get('Content-Type', '').decode().lower().startswith('text/html'):
            self.logger.info(f"Skipping the non HTML response: {response.url}")
            return

        # Extract text content from the response
        excluded_ancestors = [
            'not(ancestor::header)',
            'not(ancestor::footer)',
            'not(ancestor::input)',
            'not(ancestor::textarea)',
            'not(ancestor::button)',
        ]
        exclusion = ' and '.join(excluded_ancestors)

        selectors = [
            f'//h1[{exclusion}]/text()',
            f'//h2[{exclusion}]/text()',
            f'//h3[{exclusion}]/text()',
            f'//h4[{exclusion}]/text()',
            f'//h5[{exclusion}]/text()',
            f'//h6[{exclusion}]/text()',
            f'//p[{exclusion}]/text()',
            f'//span[{exclusion}]/text()',
            f'//div[{exclusion}]/text()',
            f'//li[{exclusion}]/text()',
            f'//article[{exclusion}]/text()',
            f'//main[{exclusion}]/text()',
        ]

        text_contents = response.xpath(' | '.join(selectors)).getall()
        text_content = ' '.join(text.strip().replace('\n', '') for text in text_contents if text.strip())

        yield {
            'url': response.url,
            'text': text_content
        }

        # Use the link extractor to find links
        links = self.link_extractor.extract_links(response)
        for link in links:
            yield response.follow(link, self.parse)

def run_scrapy_crawler(website_url):
    """
    Function to run the Scrapy crawler for a given website URL
    """

    parsed_url = urlparse(website_url)

    allowed_domains = [parsed_url.netloc]
    start_urls = [f"{parsed_url.scheme}://{parsed_url.netloc}"]

    print(f"Start crawling for URL: {parsed_url.netloc}")

    process = CrawlerProcess()

    process.crawl(WebTextSpider, allowed_domains=allowed_domains, start_urls=start_urls)
    process.start()


if __name__ == "__main__":
    run_scrapy_crawler("https://lnmiit.ac.in/")
