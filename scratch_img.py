import app as A, io, time
client = A.app.test_client()
client.post('/login', data={'email':'admin@medinsight.in','password':'admin123'}, follow_redirects=True)
# upload an existing image
img = open('uploads/20260925173559_20f59459_9.jpg','rb')
data = {'file': (io.BytesIO(img.read()), 'test_image.jpg'),
        'patient_name': 'Test Patient', 'relationship': 'Self'}
r = client.post('/upload', data=data, content_type='multipart/form-data')
print('upload status', r.status_code, r.get_data(as_text=True)[:300])
rid = r.get_json()['id']
print('report id', rid)
# wait for processing
for i in range(30):
    rep = client.get(f'/report/{rid}')
    if rep.status_code == 200:
        html = rep.get_data(as_text=True)
        if 'completed' in html or 'failed' in html or 'processing' not in html.lower():
            break
    time.sleep(1)
rep = client.get(f'/report/{rid}')
html = rep.get_data(as_text=True)
print('report status', rep.status_code)
import re
for m in re.findall(r'(processing|completed|failed|pending \d+%|\d+% )', html, re.I)[:5]:
    print(m)
print('has executive summary section:', 'Executive Summary' in html or 'WHAT THIS MEANS' in html)
print('has diagnosis:', 'Diagnosis' in html or 'Clinical Diagnosis' in html)
PY