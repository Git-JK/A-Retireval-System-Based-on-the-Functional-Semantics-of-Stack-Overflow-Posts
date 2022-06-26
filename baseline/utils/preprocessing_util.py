from nltk import word_tokenize
from bs4 import BeautifulSoup
import re


def replace_double_space(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

def tokenize_and_rebuild(sent):
    sent = str(sent).encode('utf-8', 'ignore').decode('utf-8')
    ws = word_tokenize(sent)
    return ' '.join(ws)

def remove_text_code(html_str):
    # 去除text中的code部分
    regex_pattern = r'<pre(.*?)><code>([\s\S]*?)</code></pre>'
    html_text = html_str
    for m in re.finditer(regex_pattern, html_str):
        raw_code = html_str[m.start():m.end()]
        html_text = html_text.replace(raw_code, ' ')
    return html_text.replace('\n', ' ')

def remove_html_tags(raw_html):
    # 去除text中的html tags
    try:
        text = BeautifulSoup(raw_html, "html.parser").text
    except Exception as e:
        text = clean_html_tag(raw_html)
    finally:
        return text
        
def clean_html_tag(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def preprocess_title(title):
    # 将title中的双空格转换成单空格，然后分词
    text = title.lower()
    text = replace_double_space(text.replace('\n', ' '))
    text = tokenize_and_rebuild(text)
    return text.strip()

def preprocess_body(body):
    text = body.lower()
    text = remove_text_code(text)
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    text = tokenize_and_rebuild(text)
    if text:
        return text.strip()
    else:
        return ''
    
def preprocessing_for_tag(tag_str):
    # 对post的tag处理，去掉括号并且分成[]形式
    return tag_str.replace('<', ' ').replace('>', ' ').strip().split()
    
def preprocessing_for_que(q):
    # 预处理query
    q.title = preprocess_title(q.title)
    q.body = preprocess_body(q.body)
    q.tag = preprocessing_for_tag(q.tag)
    return q
    
def preprocessing_for_ans(ans):
    # 预处理answer
    text = remove_text_code(ans.body.lower())
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    return text.strip()

def preprocessing_for_ans_sent(sent):
    text = remove_text_code(sent.lower())
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    return text.strip()


if __name__ == "__main__":
    text = 'text > <img src= http://i.stack.imgur.com/ltCod.    png alt= Rich task editor >   '
    print(remove_html_tags(text))
    print(replace_double_space(text))
