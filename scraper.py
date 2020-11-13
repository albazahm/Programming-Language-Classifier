import requests
import bs4
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import pandas as pd
import re
from multiprocessing import Pool, Manager
import pickle
import csv
import json

MAIN_URL = "http://www.rosettacode.org"
TASKS_URL = "http://www.rosettacode.org/wiki/Category:Programming_Tasks"

class RosettaCodeScraper():
    
    def __init__(self, main_url, tasks_url, output_file='RosettaCodeData.csv'):

        """
        Initialize the RosettaCodeScraper class. Creates a file in which to store the scraped data
        and adds headings to it.

        Parameters
        ----------
        main_url: str
            the homepage URL for RosettaCode
        tasks_url: str
            the URL where the tasks are listed at RosettaCode
        languages: list
            list of programming languages to scrape
        output_file: str='RosettaCodeData.csv'
            filename to store output
        """
        
        self.main_url = main_url
        self.tasks_url = tasks_url
        self.output = dict()
        self.count = 0
        self.output_file = output_file
        with open('./languages.json') as f:
            self.languages = dict(json.load(f))
        self.languages = set(self.languages['languages'])
        print(f'Registered {len(self.languages)} languages...') 
        
        #create output file with headers
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['language', 'code'])
    
    def get_paths(self):

        """
        Function to get the path to each task in the RosettaCode website

        Returns
        -------
        paths: list
            list of paths to each task in the RosettaCode website
        """
        #initialize empty list of paths to access
        paths = list()
        #request task page
        try:
            page = requests.get(self.tasks_url)
            #parse page with html parser
            soup = BeautifulSoup(page.content, 'html.parser')
            #find div with all the task links
            body = soup.find('div', id='mw-pages')
            #extract links and append to list
            links = body.find_all('a')
        except OSError:
            print('page does not exist')
        
        #append links to paths list
        for link in links:
            sublink = link.get('href')
            paths.append(sublink)

        self.paths = paths
        return paths
    
    def scrape(self, path):

        """
        Function to scrape data given a path
        """
        #create URL using the main URL and the task URL
        url = self.main_url + path

        try:
            #request task page
            page = requests.get(url)
            #parse page with html5lib
            soup = BeautifulSoup(page.content, 'html5lib')
            #find div containing the code for all languages
            body = soup.find('div', id='mw-content-text')
            #isolate the pre-formatted text tags containing the code
            tasks = body.find_all('pre', class_=re.compile(r'.*highlighted_source'))
        except:
            return None

        #loop over tasks
        for task in tasks:
            try:
                #get coding language name from h2 tag
                name = task.find_previous('h2').text[:-6]
            except:
                continue
            
            if name not in self.languages:
                continue
            
            try:
                #ignore snippets that correspond to outputs
                header = task.previousSibling.previousSibling.text.lower()
                if ('output' in header) and (len(header)<10):
                    continue
            except:
                continue
            #intialize list of outputs
            output_list = []
            for t in task:
            #append each component of the pre tags to the output
                if type(t)==NavigableString:
                    output_list.append(t)
                else:
                    if t.name=='br':
                        output_list.append(t.name)
                    else:
                        output_list.append(t.text)
            #join components to one string
            output = ' '.join(output_list)
            #replace commas to complex string (csv format output will be interpret commas in code as newline which we don't want)
            output = output.replace(',', '!@#$%^&&^%$#@!')
            
            #write output to file
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([name, output])
            

            self.count += 1
            if self.count%100==0:
                print(f'100 snippets loaded...')

if __name__ == '__main__':

    print('Initializing Scraper...')
    scraper = RosettaCodeScraper(main_url=MAIN_URL, 
                                 tasks_url=TASKS_URL,
                                #  languages=LANGUAGES
                                 )
    
    print('Obtaining Paths...')
    paths = scraper.get_paths()
    print('Scraping...')

    
    p = Pool(processes=8)
    p.map(scraper.scrape, paths)
    p.terminate()
    p.join()
    print('Completed...')
