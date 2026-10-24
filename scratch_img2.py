import app as A, io, time
client = A.app.test_client()
client.post('/login', data={'email':'admin@medinsight.in','password':'admin123'}, follow_redirects=True)
img = open('uploads/20260925173559_20f59459_9.jpg','rb')
data = {'file': (io.BytesIO(img.read()), 'test_image.jpg'),
        'patient_name': 'Test Patient', 'relationship': 'Self'}
r = client.post('/upload', data=data, content_type='multipart/form-data')
print('upload status', r.status_code, r.get_data(as_text=True)[:200])
rid = r.get_json()['id']
print('report id', rid)
status=None
for i in range(60):
    rr = client.get(f'/api/report_status/{rid}')
    try:
        j=rr.get_json(); status=j.get('status'); prog=j.get('progress')
    except Exception:
        status=None
    if status in ('completed','failed'):
        break
    time.sleep(1)
print('final status', status, 'progress', prog)
rep = client.get(f'/report/{rid}')
html = rep.get_data(as_text=True)
print('report status', rep.status_code)
print('has Executive Summary:', 'Executive Summary' in html)
print('has Clinical Diagnosis:', 'Clinical Diagnosis' in html)
print('has Leg Fracture content:', 'fracture' in html.lower() or 'bone' in html.lower())
print('has Marathi/Hindi picker:', 'Marathi' in html and 'Hindi' in html)
print('has translateBtn:', 'id="translateBtn"' in html)
PY