import mechanicalsoup
import pandas as pd
from bs4 import BeautifulSoup as bs

# setup

browser = mechanicalsoup.StatefulBrowser(
    soup_config={'features': 'lxml'},
    raise_on_404=True,
    user_agent='MyBot/0.1: mysite.example.com/bot_info',
)

browser.open("https://forums.somethingawful.com/account.php?action=loginform")
form = browser.select_form('form[class="login_form"]')
form.set("username", "dprk -i juche.deb")
form.set("password", "Notahim3>someL")
resp = browser.submit_selected()


def pnames(parent_list):
    return [p.name for p in parent_list]


def scrape_thread(threadid: int, pages: int):
    all_posts = pd.DataFrame()
    for i in range(1, pages+1):
        url = "https://forums.somethingawful.com/showthread.php?threadid=" + \
            str(threadid) + "&perpage=40&pagenumber=" + str(i)
        page = browser.get(url)
        soup = bs(page.text, 'html.parser')

        posts = soup.find_all(attrs={'class': 'postbody'})
        authors = soup.find_all(attrs={'class': 'author'})
        dates = soup.find_all(attrs={'class': 'postdate'})

        cleaned_posts = []
        for p in posts:
            filtered_post = ''.join(text for text in p.find_all(text=True)
                                    if "blockquote" not in pnames(text.parents)
                                    and
                                    "h4" not in pnames(text.parents)
                                    and
                                    "google" not in text
                                    and
                                    "fucked around with this post" not in text
                                    and
                                    "(USER WAS" not in text
                                    )
            stripped = filtered_post.strip().replace('\n', ' ')
            cleaned_posts.append(stripped)

        cleaned_authors = [a.text for a in authors]
        cleaned_dates = [d.text.strip(' #\n?') for d in dates]

        d = {"author": cleaned_authors, "post": cleaned_posts, "date":
             cleaned_dates}
        df = pd.DataFrame(data=d)

        all_posts = all_posts.append(df)
    return all_posts.reset_index(drop=True)


def scrape_and_save(filename, threadid, pages):
    thread = scrape_thread(threadid, pages)
    thread.to_csv(filename, index=False)


# scrape 300 pages of each thread and save to csv files
scrape_and_save("cspam_climate.csv", 3884239, 300)
scrape_and_save("dnd_climate.csv", 3874548, 300)
scrape_and_save("cspam_primary.csv", 3903588, 300)
scrape_and_save("dnd_primary.csv", 3898642, 300)
