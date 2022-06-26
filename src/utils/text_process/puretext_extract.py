import re
from bs4 import BeautifulSoup
from nltk import word_tokenize


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

def clean_html_tag(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def remove_html_tags(raw_html):
    # 去除text中的html tags
    try:
        text = BeautifulSoup(raw_html, "html.parser").text
    except Exception as e:
        text = clean_html_tag(raw_html)
    finally:
        return text

def remove_html_line_feed(text):
    while '&#xA;' in text:
        text = text.replace('&#xA;', ' ')
    return text        

def puretext_extract(text):
    text = text.lower()
    text = remove_text_code(text)
    text = remove_html_tags(text)
    text = replace_double_space(text.replace('\n', ' '))
    text = remove_html_line_feed(text)
    return text

def remove_stopwords(sent, sw):
    if isinstance(sent, str):
        wlist = word_tokenize(sent)
    elif isinstance(sent, list):
        wlist = sent
    else:
        raise Exception('Wrong type for removing stopwords!')
    sent_words = []
    for w in wlist:
        if w == '':
            continue
        if w not in sw:
            sent_words.append(w)
    return sent_words

def preprocess_title(title):
    text = title.lower()
    text = replace_double_space(text.replace('\n', ' '))
    text = tokenize_and_rebuild(text)
    return text.strip()
    
if __name__ == "__main__":
    test_text = "<p>To me, a singleton makes sense wherever you want to represent something which is unique in its kind. </p>&#xA;&#xA;<p>As an example, if we wanted to model <em>the</em> <code>Sun</code>, it could not be a normal class, because there is only one <code>Sun</code>. However it makes sense to make it inherit from a <code>Star</code> class. In this case I would opt for a static instance, with a static getter.</p>&#xA;&#xA;<p>To clarify, here is what I'm talking about :</p>&#xA;&#xA;<pre><code>public class Star {&#xA;    private final String name;&#xA;    private final double density, massInKg;  &#xA;&#xA;    public Star(String name, double density, double massInKg) {&#xA;        // ...&#xA;    }&#xA;&#xA;    public void explode() {&#xA;       // ...&#xA;    }&#xA;}&#xA;&#xA;public final class Sun extends Star {&#xA;    public static final Sun INSTANCE = new Sun();&#xA;&#xA;    private Sun() { super(\"The shiniest of all\", /**...**/, /**...**/); }&#xA;}&#xA;</code></pre>&#xA;&#xA;<p><code>Sun</code> can use all the methods of <code>Star</code> and define new ones. This would not be possible with an enum (extending a class, I mean).</p>&#xA;&#xA;<p>If there is no need to model this kind of inheritance relationships, as you said, the <code>enum</code> becomes better suited, or at least easier and clearer. For example, if an application has a single <code>ApplicationContext</code> per JVM, it makes sense to have it as a singleton and it usually doesn't require to inherit from anything or to be extendable. I would then use an <code>enum</code>.</p>&#xA;&#xA;<p>Note that in some languages such as Scala, there is a special keyword for singletons (<code>object</code>) which not only enables to easily define singletons but also completely replaces the notion of static method or field.</p>&#xA;"
    print(puretext_extract(test_text))
    
    