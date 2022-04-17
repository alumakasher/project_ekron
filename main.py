from group_class import FacebookGroups
from scraper import scrape
from db_handler import init_mongo
from datetime import datetime
from rich import print
import config
import time
import random
import os
import sys


# Sleep for the desired time and restart for the sake of establishing new requests session
def sleep_and_restart(time_off):
    time.sleep(time_off)
    os.execl(sys.executable, 'python', __file__, *sys.argv[1:])


# Running instances, the main function determines which group will be scraped and when.
# When a group scraping terminates, main will decide what happens next, depending on the cause
def main():

    init_mongo()
    list_size = len(FacebookGroups.reg_list)
    i = 0
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"\n[green]{dt_string} Scraper initiated")
    while True:

        print(f"\n[green]Group Num: {i+1}\nGroup name: {FacebookGroups.reg_list[i % list_size].name}")
        kwargs = FacebookGroups.reg_list[i % list_size].__dict__
        state = scrape(**kwargs)
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if state == 0:
            print(f"\n[green]{dt_string} Breaking after reaching max number of posts per batch. "
                  f"Scraping of next batch will begin in {str(config.BREAK_TIME/60/60)} hrs")
            FacebookGroups.batch_posts = 0
            sleep_and_restart(config.BREAK_TIME)
        elif state == 1:
            print(f"\n[green]{dt_string} Scraping of {FacebookGroups.reg_list[i % list_size].name} terminated after it's"
                  f" gone through {FacebookGroups.reg_list[i % len(FacebookGroups.reg_list)].max_known} known posts")
        elif state == 2:
            print(f"\n[red]{dt_string} Temporarily banned by Facebook. "
                  f"Scraping of next batch will begin in {str(config.BAN_SLEEP/60/60)} hrs")
            sleep_and_restart(config.BAN_SLEEP)

        i += 1
        if i % list_size == 0:
            print(f"\n[green]{dt_string} Breaking after covering all groups. "
                  f"Scraping of next batch will begin in {str(config.BREAK_TIME/60/60)} hrs")
            sleep_and_restart(config.BREAK_TIME)


'''
#### START HERE #####
Add group instances, mandatory attributes are group name and group id
Unless specified, all other attributes will be set as configured in config.py
'''
if __name__ == '__main__':

    # These are example public groups:
    jackson_heights = FacebookGroups('Jackson Heights', 35851988964)
    ilsington = FacebookGroups('Ilsington', 743308202528264)
    uppereast = FacebookGroups('Upper East Side', 'uppereastside35andolder')

    main()





