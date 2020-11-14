class ReviewParser:
    def __init__(self, url, file_path):
        self.url = url
        self.file_path = file_path

    def parse(self):
        import urllib.request
        from inscriptis import get_text

        html = urllib.request.urlopen(self.url).read().decode('utf-8')
        text = get_text(html)

        try:
            with open(self.file_path, 'w') as f:
                f.write(text)
        finally:
            return text
