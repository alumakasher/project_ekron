from facebook_scraper import get_posts
from facebook_scraper import exceptions
from group_class import FacebookGroups
from datetime import datetime
from db_handler import handle_post
from db_handler import init_collection
import config
import db_handler
import random
import time


# Run scraper and pass all posts to db_handler
def run(**kwargs):

    try:

        posts = get_posts(group=kwargs['id'], cookies=kwargs['cookies'], pages=10, timeout=40,
                          options={"comments": kwargs['comments'], "allow_extra_requests": False})

        known_count = 0
        for post in posts:

            if not post['post_id']:
                continue

            if kwargs['comments']:
                if post['comments_full']:
                    comments = post['comments_full']
                    for comment in comments:
                        time.sleep(random.uniform(1.5, 3))
                        if comment['replies']:
                            replies = comment['replies']
                            for reply in replies:
                                time.sleep(random.uniform(1.5, 3))

            FacebookGroups.batch_posts += 1
            dt_string = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
            print(f"\nPost num {FacebookGroups.batch_posts} for this batch in group "
                  f"'{kwargs['name']}', received on {dt_string} :\n")
            for k in post.keys():
                if k in config.POST_ATTRIBUTES and post[k]:
                    text = str(post[k]).replace("\n", " ")
                    if len(text) > 90:
                        text = text[:90] + "..."
                    print(f"{k}: {text.replace('  ', ' ').replace(' ...', '...')}")

            if FacebookGroups.batch_posts == config.MAX_NEW_POSTS:
                handle_post(post)
                return 0

            if handle_post(post):
                known_count += 1
                print(f"New post: No. Occurrence {str(known_count)}# in a row")
                if known_count == kwargs['max_known']:
                    return 1
            else:
                known_count = 0
                print("New post: Yes")

            time.sleep(random.uniform(config.POSTS_GAP[0], config.POSTS_GAP[1]))

    except exceptions.TemporarilyBanned:
        return 2


# Initiate a MongoDB collection and run scraper
def scrape(**kwargs):

    init_collection(kwargs['name'], kwargs['id'])

    return run(**kwargs)
