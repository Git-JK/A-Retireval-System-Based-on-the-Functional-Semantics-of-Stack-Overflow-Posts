
def get_html_score(text):
    HTMLTAG_plus = ['<strong>', '<code>']
    HTMLTAG_minus = ['<strike>']
    len_of_tag_plus = len(HTMLTAG_plus)
    score = 1.0
    for pattern in HTMLTAG_plus:
        if pattern in text.lower():
            score += (1.0 / len_of_tag_plus)
    for pattern in HTMLTAG_minus:
        if pattern in text.lower():
            score = 1.0
            break
    return score

