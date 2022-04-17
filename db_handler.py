from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import config
import os


# Initiate cluster and db
def init_mongo():

    load_dotenv()
    mongo_uri = os.environ['MONGODB_URI']
    global db
    client = MongoClient(mongo_uri)
    db = client["group_scraper_new"]


def init_collection(name, _id):

    global collection
    collection = db[name]

    group_dict = collection.find_one({'post_id': "0"})
    if not group_dict:
        collection.insert_one({'post_id': "0", 'group_id': str(_id)})


# When scarping with comments, remove unneeded fields
def comments_cleanup(comments):

    new_comments = []

    for comment in comments:

        new_comment = {}
        for key in comment:
            if key in config.COMMENT_ATTRIBUTES:
                new_comment[key] = comment[key]
                continue
            if key == "replies" and comment['replies']:
                new_comment['replies'] = comments_cleanup(comment['replies'])
        new_comments.append(new_comment)

    return new_comments


# If post is known to the DB, update it and save older information in an array
def post_known(scraped_post, db_post):

    new_doc = {}
    for key, val in scraped_post.items():
        if key not in db_post:
            if val and key in config.POST_ATTRIBUTES:
                new_doc[key] = val
        else:
            if val:
                new_doc[key] = val

    new_doc['label'] = db_post['label']
    new_doc['update_time'] = datetime.now()
    collection.replace_one({'post_id': scraped_post['post_id']}, new_doc)

    return 1


# If post is unknown to the DB, insert the post with the desirable attributes
def insert_post(post):

    inserted_post = {}

    for key, val in post.items():
        if val and key in config.POST_ATTRIBUTES:
            inserted_post[key] = val

    inserted_post['label'] = "Not Yet Reviewed"
    inserted_post['update_time'] = datetime.now()
    collection.insert_one(inserted_post)
    return 0


# Main function of db_handler
# Receives the scraped post and checks if it is a known post to DB
def handle_post(scraped_post):

    if not scraped_post['post_id']:
        return 0

    if scraped_post['comments_full']:
        scraped_post['comments_full'] = comments_cleanup(scraped_post['comments_full'])

    db_post = collection.find_one({'post_id': scraped_post['post_id']})

    if db_post:
        if db_post['label'] == "1":
            return 1
        difference = (datetime.now() - db_post['update_time'])
        if difference.total_seconds() < 60:
            return 0
        del db_post['_id']
        return post_known(scraped_post, db_post)

    return insert_post(scraped_post)
