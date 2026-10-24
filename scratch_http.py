import urllib.request, urllib.parse, http.cookiejar, json, time, os

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
BASE='http://127.0.0.1:5000'

def post(url, data=None, headers=None, files=None):
    if files is not None:
        boundary = '----WebKitFormBoundary'+os.urandom(8).hex()
        parts=[]
        for k,v in data.items():
            parts.append(f'--{boundary}'.encode())
            parts.append(f'Content-Disposition: form-data; name="{k}"'.encode())
            parts.append(b'')
            parts.append(v.encode() if isinstance(v,str) else v)
        for k,(fname,fbytes,ctype) in files.items():
            parts.append(f'--{boundary}'.encode())
            parts.append(f'Content-Disposition: form-data; name="{k}"; filename="{fname}"'.encode())
            parts.append(f'Content-Type: {ctype}'.encode())
            parts.append(b'')
            parts.append(fbytes)
        parts.append(f'--{boundary}--'.encode())
        parts.append(b'')
        body=b'\r\n'.join(parts)
        h={'Content-Type': f'multipart/form-data; boundary={boundary}'}
        if headers: h.update(headers)
        req=urllib.request.Request(BASE+url, data=body, headers=h, method='POST')
    else:
        body=urllib.parse.urlencode(data).encode() if data else None
        h={'Content-Type':'application/x-www-form-urlencoded'}
        if headers: h.update(headers)
        req=urllib.request.Request(BASE+url, data=body, headers=h, method='POST')
    try:
        resp=opener.open(req, timeout=30)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

# login
s,b=post('/login', {'email':'admin@medinsight.in','password':'admin123'})
print('login', s)

# upload image
img=open('uploads/20260925173559_20f59459_9.jpg','rb').read()
s,b=post('/upload', data={'patient_name':'Test Patient','relationship':'Self'},
         files={'file':('test_image.jpg', img, 'image/jpeg')})
print('upload', s, b[:200])
rid=json.loads(b)['id']
print('report id', rid)

# poll status
status=None
for i in range(90):
    s,b=opener.open(BASE+f'/api/report_status/{rid}', timeout=10).read(), None
    try:
        j=json.loads(s); status=j.get('status'); prog=j.get('progress')
    except Exception:
        status=None
    if status in ('completed','failed'):
        break
    time.sleep(1)
print('final status', status, 'progress', prog)

# fetch report page
s,b=opener.open(BASE+f'/report/{rid}', timeout=10), None
html=s.read().decode()
print('report page', s.status)
print('has Clinical Diagnosis:', 'Clinical Diagnosis' in html)
print('has Executive Summary:', 'Executive Summary' in html)
print('has Marathi picker:', 'Marathi' in html)
print('has translateBtn:', 'id="translateBtn"' in html)

# download pdf
s,b=opener.open(BASE+f'/report/{rid}/download.pdf', timeout=30), None
pdf=s.read()
print('pdf status', s.status, 'content-type', s.headers.get('Content-Type'), 'bytes', len(pdf), 'isPDF', pdf[:4]==b'%PDF')
open('MedInsight_Report_'+str(rid)+'.pdf','wb').write(pdf)
print('saved pdf')
