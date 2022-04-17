# Project-Ekron

This is a python based tool intended to integrate as an extension to a smart city management system. 
The tool identifies relevant content from social networks using machine learning models.

Social networks contain plenty of useful information for municipalities and other regional councils for running their city/region. But detecting such information in this nonstopping and huge information stream can be like finding a needle in a haystack. Therefore, a tool that can detect, point and save such information can be very beneficial.

See requierments file.
In addition, to get the tool started you are going to need:
1. A .env file containing a connection string to your MongoDB cluster
2. A text file named "cookies.txt" containing browser cookies after your facebook login. For that, use browser extensions:

* [Get Cookies.txt (Chrome)](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid?hl=en) 
* [Cookie Quick Manager (Firefox)](https://addons.mozilla.org/en-US/firefox/addon/cookie-quick-manager/)

Make sure that you include both the c_user cookie and the xs cookie.

Instructions:
- In main.py, add FacebookGroup class objects containing desired groups name and groups id.
- Run listener.py for the detection of content
- Once listener is up and running, run main.py for scraping to begin





