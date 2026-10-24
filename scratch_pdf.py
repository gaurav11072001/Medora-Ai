import app as A, os
client = A.app.test_client()
client.post('/login', data={'email':'admin@medinsight.in','password':'admin123'}, follow_redirects=True)
r = client.get('/report/1/download.pdf')
print('status', r.status_code)
print('content-type', r.content_type)
data = r.get_data()
print('bytes', len(data))
print('startswith pdf', data[:4] == b'%PDF')
out = 'MedInsight_Report_1_check.pdf'
with open(out,'wb') as f: f.write(data)
print('saved', out, os.path.getsize(out))
PY