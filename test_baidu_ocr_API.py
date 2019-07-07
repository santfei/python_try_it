import urllib.request
import urllib.parse
import json

ak = 'T7aymXY6VzOvrv5nvlVt4ltG'
sk = 'UniRi2IIjhXLGMTVFghWtYsNOinzc6SA'
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s'%(ak, sk)

request = urllib.request.Request(host)
request.add_header('Content-Type', 'application/json; charset=UTF-8')

response = urllib.request.urlopen(request)
content = response.read()
if (content):
    print(content)

print('\n\n')
json_all = json.loads(content)
access_token = json_all['access_token']
print(json_all)

url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token=%s'% access_token
data = urllib.parse.urlencode({'url': 'https://b-ssl.duitang.com/uploads/item/201702/17/20170217212539_dxKe2.jpeg'}).encode()
req = urllib.request.Request(url, method='POST')
req.add_header('Content-Type','application/x-www-form-urlencoded')
res = urllib.request.urlopen(req, data).read().decode('utf8')
ocr = json.loads(res)
print(ocr)
for item in ocr['words_result']:
    print(item['words'])