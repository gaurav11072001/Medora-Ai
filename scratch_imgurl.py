import urllib.request, urllib.parse, http.cookiejar
cj=http.cookiejar.CookieJar(); o=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
# login
body=urllib.parse.urlencode({'email':'admin@medinsight.in','password':'admin123'}).encode()
req=urllib.request.Request('http://127.0.0.1:5000/login', data=body, headers={'Content-Type':'application/x-www-form-urlencoded'}, method='POST')
o.open(req, timeout=15)
# fetch image
r=o.open('http://127.0.0.1:5000/report_file/4', timeout=15)
print('image status', r.status, 'content-type', r.headers.get('Content-Type'), 'bytes', len(r.read()))
# fetch report page
rp=o.open('http://127.0.0.1:5000/report/4', timeout=15)
html=rp.read().decode()
print('report page', rp.status)
print('has img tag:', '<img' in html and 'report_file/4' in html)
print('has open link:', 'Open original file' in html)
