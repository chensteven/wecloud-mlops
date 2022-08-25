'''
scrape_example.py

Enter in a number of pages to scrape and this program will generate a list of hackernews articles
with over 100 votes
'''
import pprint
import scrape
import json
NUM_PAGES = 439
FIRST_PAGE_LINK = 'https://www.datascienceweekly.org/newsletter/data-science-weekly-newsletter-issue'

page_links = scrape.scrape_methods.get_page_links(FIRST_PAGE_LINK, NUM_PAGES)
mega_links_list = scrape.scrape_methods.create_mega_link_list(page_links)

with open('../data/dsweekly_data.json', 'w') as f:
  json.dump(mega_links_list, f)