import requests
from lxml import html

base_url = 'https://www.tiobe.com/tiobe-index/'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
}

response = requests.get(base_url, headers=headers)

document = html.fromstring(response.text)

th_list = document.xpath("//*[@id='top20']/thead/tr/th/text()")
print(th_list)

tr_list = document.xpath("//*[@id='top20']/tbody/tr")

for tr in tr_list:
    td = tr.xpath("./td/text()")
    print(td)