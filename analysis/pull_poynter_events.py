#!/usr/bin/python3

"""
Downloads articles from fact-check articles in our Poynter database,
and extracts the external links in them, for building up a Twitter TDT
training set.
"""

import csv
import json

from bs4 import BeautifulSoup
from newspaper import Article
import tldextract

# Extract events if the Poynter article for the event is one of these domains
_ALLOWED_ARTICLE_DOMAINS = ["politifact.com"]

# Do not extract links to these domains.  These are deemed unlikely to represent
# information that is specific to the event.
_DISALLOWED_LINK_DOMAINS = ["facebook.com", "twitter.com", "nih.gov", "who.int"]

_ARTICLE_FIELD = " Article Link"

def domain_of_url(url):
    """Given a url, return the top-level domain, like nytimes.com."""
    domain_info = tldextract.extract(url)
    if not domain_info:
        return None
    return domain_info.domain + "." + domain_info.suffix

def main():
    """Read the poynter data file and write out json about the articles."""
    with open("poynter_data.csv") as fs_poynter:
        reader = csv.DictReader(fs_poynter, delimiter=";")
        for row in reader:
            if _ARTICLE_FIELD not in row or row[_ARTICLE_FIELD] is None:
                continue
            article_url = row[" Article Link"].strip()
            article_domain = domain_of_url(article_url)

            if article_domain in _ALLOWED_ARTICLE_DOMAINS:
                # Download the article with newspaper3k
                article = Article(article_url)
                article.download()
                article.parse()

                # Get links from article body using BeautifulSoup
                html = article.html
                soup = BeautifulSoup(html, 'html.parser')

                # Get the URLs in article body
                urls = set()
                for link in soup.find_all('a'):
                    url = link.get("href")
                    if url and url.startswith("http"):
                        domain = domain_of_url(url)
                        if domain and domain not in _DISALLOWED_LINK_DOMAINS:
                            # External link
                            urls.add(url)

                # Dump the data for this event as json
                print(json.dumps(
                    {"url": article_url,
                     "urls_in_body": list(urls),
                     "text": article.text,
                     "date": row["Date"],
                     "false_statement": row.get(" False Statement", ""),
                     "true_statement": row.get(" True Statement", "")}))


if __name__ == '__main__':
    main()
