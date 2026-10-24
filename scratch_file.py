import app as A
client = A.app.test_client()
client.post('/login', data={'email':'admin@medinsight.in','password':'admin123'}, follow_redirects=True)
r = client.get('/report/4/file')
print('report_file/4 status', r.status_code, 'content-type', r.content_type, 'bytes', len(r.data), 'isjpg', r.data[:3]==b'\xff\xd8\xff')
PY