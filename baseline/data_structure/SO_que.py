class SO_Que:
    __slots__ = 'id', 'title', 'body', 'tag', 'title_words', 'matrix', 'idf_vector'
    def __init__(self, id, title, body, tag):
        self.id = id
        self.title = title
        self.body = body
        self.tag = tag