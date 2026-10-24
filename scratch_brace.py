import re
html=open('templates/report.html').read()
m=re.search(r'<script>(.*?)</script>', html, re.DOTALL)
js=m.group(1)
print('translate def present:', 'async function translate' in js)
print('braces { }:', js.count('{'), js.count('}'))
print('parens ( ):', js.count('('), js.count(')'))
print('brackets [ ]:', js.count('['), js.count(']'))
# find first syntax-ish issue: count braces as we scan
bal=0
for i,ch in enumerate(js):
    if ch=='{': bal+=1
    elif ch=='}': bal-=1
    if bal<0:
        print('NEGATIVE brace at', i, js[max(0,i-40):i+40]); break
print('final brace balance:', bal)
