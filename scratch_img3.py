import io, time, os
from dotenv import load_dotenv
load_dotenv()
from ai_engine import extract_text_from_file, analyze_document
path='uploads/20260925173559_20f59459_9.jpg'
t=time.time()
try:
    text=extract_text_from_file(path, 'test_image.jpg')
    print('EXTRACT len', len(text or ''), 'elapsed', round(time.time()-t,1),'s')
    print('TEXT PREVIEW:', (text or '')[:300])
    t=time.time()
    result=analyze_document(text, 'test_image.jpg', image_path=path, rag_context='', entities={})
    print('ANALYZE OK elapsed', round(time.time()-t,1),'s')
    print('keys', list(result.keys()))
    print('execSummary', result.get('executiveSummary','')[:200])
except Exception as e:
    import traceback; traceback.print_exc()
PY