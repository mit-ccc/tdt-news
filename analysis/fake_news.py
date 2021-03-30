from bs4 import BeautifulSoup
import urllib.request,sys,time
from urllib.request import Request, urlopen
import requests
import pandas as pd
import csv

def get_html(url):
    try:
        req = Request(url, headers = {'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        page_soup = BeautifulSoup(webpage, "html.parser")
    except:
        print("Error with parsing:", url)
    
    return page_soup

def get_pages(url):
    page_links = []
    for i in range(1, 810):
        page_links.append(url + 'page/' + str(i) + '/')
    return page_links

def get_article_page_links(url):
    links = set()
    try:
        page_html = get_html(url)
        articles = page_html.findAll("article")
    except:
        print("Error with parsing:", url)
        return []
    for article in articles:
        try:
            for link in article.select("a"):
                links.add(link['href'])
        except:
            continue
    return links

def get_true_false_info(url):
    false_phrase = ''
    true_phrase = ''
    count = 1
    article_true_link = ''
    
    try:
        page_soup = get_html(url)
        false = page_soup.findAll("h1", {"class": "entry-title"})
        true = page_soup.findAll("p", {"class": "entry-content__text entry-content__text--explanation"})
        article_true_link = page_soup.find("a", {"class": "button entry-content__button entry-content__button--smaller"}).get('href')
        date = str(page_soup.find("p", {"class": "entry-content__text entry-content__text--topinfo"}))[68:78]
    except:
        print("Error with url:", url)
    
    try:
        for phrase in false[0]:
            if count == 3:
                false_phrase = phrase
            count += 1
    except:
        print('Error with false phrase')
        return []
    
    try:
        count = 1
        for phrase in true[0]:
            if count == 1:
                true_phrase = phrase
            count += 1
    except:
        print('Error with true phrase')
        return []
    
    return [false_phrase[1:], true_phrase[13:], article_true_link, date]

def get_statements(url):
    pages = get_pages(url)
    article_page_links = set()
    page_count = 1
    for page in pages:
        print("This is page", page_count)
        article_page_links.update(get_article_page_links(page))
        page_count += 1
    true_false = dict()
    link_count = 1
    for link in article_page_links:
        print("This is link", link_count)
        true_false[link] = get_true_false_info(link)
        link_count += 1
    return true_false

def get_false_statements(true_false):
    false = []
    for statements in true_false.values():
        false.append(statements[0])
    return false

def get_true_statements(true_false):
    true = []
    for statements in true_false.values():
        true.append(statements[1])
    return true

def save_csv(save_dict, name):
    with open(name, 'w') as f:
        f.write("Date; Article Link; False Statement; True Statement;\n")
        for key in save_dict.keys():
            f.write("%s; %s; %s; %s\n" % (save_dict[key][3], save_dict[key][2], save_dict[key][0], save_dict[key][1]))

if __name__ == "__main__":
    home_page_url = 'https://www.poynter.org/ifcn-covid-19-misinformation/'
    statement_dict = get_statements(home_page_url)
    save_csv(statement_dict, "poynter_data.csv")