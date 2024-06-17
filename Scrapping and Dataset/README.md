# Web Scrapping
Web scraping is the process of extracting information from websites, web pages, or online documents. In this case, web scraping is used to extract information about articles from the site [earth911.com](https://www.earth911.com) using the BeautifulSoup library. The article information is then stored in a Python DataFrame. The content of the articles in this DataFrame will later be used to calculate TF-IDF and cosine similarity when the recommendation system is applied.

# Keypoints
- Web scraping using BeautifulSoup library
- Extracting article information from earth911.com
- Storing article information in a Python DataFrame
- List of articles that will be scrapped is stored in xlsx format. Pandas read_excel method is used to access article links

# Step by step to do the scraping
1. **Install necessary libraries**: You need to install `beautifulsoup4`,`requests`,`numpy` and `pandas`
    ```python 
    # Library Import
    import pandas as pd
    import numpy as np
    import requests
    from bs4 import BeautifulSoup
    import re
    ```
2. **Create empty DataFrame contains information needed** : Create DataFrame using pandas method with columns that contain information needed.
    ```python 
    df = pd.DataFrame(columns=['title','raw_content','clean_content','date_created','author','imageSrc','articleLink'])
    ```
3. **Create methods that clean text as needed** : This is an optional step where you can create a useful method function to clean text so that the stored information can be cleaner.
    ```python
    def clean_text(text):
        text = text.replace('\n',' ').strip()
        text = re.sub(' +',' ',text)
        return text
    ```
4. **Read article links that stored in xlsx format**: The dataframe in the xlsx format stored information about article title and article link. Pandas read_excel is used to read the dataframe.
    ```python
    list_artikel = pd.read_excel('/content/LinkArtikelEnglish.xlsx')
    ```

5. **Iterates over row and access each article links using requests library**: Use Requests library to access each article link. After that, we can scrap relevant information on the webpage using beautifulsoup libraries. Sample code is shown below.
    ```python
    #Iterates over links to gather information
    for index,row in list_artikel.iterrows():
        url = row['Link Artikel']
        response = requests.get(url,headers={'User-agent':'Reyhan'})
        soup = BeautifulSoup(response.text, 'html.parser')

    #Scrapped the article based on information needed
        title = soup.find('h1',class_='title').text
        title = clean_text(title)
        raw_content = soup.find('article',class_='small single').text.strip()
        clean_content = clean_text(raw_content)
        date_created = soup.find('span',class_='mg-blog-date').text.strip()
        author = soup.find('h4',class_='media-heading').text.strip()
        author = author.replace('By','')
        image = soup.find('img',class_='img-fluid wp-post-image')['src']
        link = url
    ```

6. **Store scrapped information into DataFrame**: After scrapped the information, we can store it into DataFrame
    ```python
    df.loc[index,'title'] = title
    df.loc[index,'raw_content'] = raw_content
    df.loc[index,'clean_content'] = clean_content
    df.loc[index,'date_created'] = date_created
    df.loc[index,'author'] = author
    df.loc[index,'imageSrc'] = image
    df.loc[index,'articleLink'] = link
    ```