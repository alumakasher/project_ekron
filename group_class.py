import config


class FacebookGroups:

    reg_list = []
    batch_posts = 0

    def __init__(self, group_name, group_id, cookies=config.COOKIES,
                 max_known=config.MAX_KNOWN_POSTS, comments=config.COMMENTS):
        self.name = group_name
        self.id = group_id
        self.cookies = cookies
        self.max_known = max_known
        self.comments = comments
        FacebookGroups.reg_list.append(self)















