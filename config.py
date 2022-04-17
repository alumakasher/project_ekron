
# Set this to be the number of posts scraped and known to the DB before scraping of group terminates
# For lower likelihood of missing new and relevant posts set a higher number
MAX_KNOWN_POSTS = 5

# Set this to be the general number of posts scraped before scraping of batch terminates
# For lower risk of account ban set a lower number
MAX_NEW_POSTS = 50

# Number of seconds program waits before running next batch
# For lower risk of account ban set a higher number
BREAK_TIME = 60*60*0.6

# Number of seconds program waits when it gets temporarily banned by Facebook
BAN_SLEEP = 60*60*24

# Range of seconds scraper sleeps after each post is implemented in the DB (Random)
POSTS_GAP = (35, 45)

# List of post attributes that will be inserted into the database by db_handler
POST_ATTRIBUTES = ['post_id', 'text', 'time', 'user_id', 'username', 'images_description', 'shared_text',
                   'shared_post_id', 'shared_user_id', 'shared_username', 'comments_full']

# List of comment attributes that will be inserted into the database by db_handler
COMMENT_ATTRIBUTES = ['comment_id', 'commenter_id', 'commenter_name', 'comment_text']

# Default cookies file. Useful when only one cookies file is being used
COOKIES = 'cookies.txt'

# Default configuration for scraping with/without comments
COMMENTS = False

