import pathlib


class RawReview:
    def __init__(self, author, date_time, text):
        self.author = author
        self.date_time = date_time
        self.text = text


def get_reviews_from_raw_onliner(file_path):
    paras = open(file_path).read().split("\n")

    import re
    date_pat = r'\d{2}.\d{2}.\d{4}\sв\s\d{2}:\d{2}'
    reviews = []
    i = 0
    for par in paras:
        if re.match(date_pat, par):
            date = par.replace("в", " ").replace("   ", "_").replace(":", ".")
            rv = RawReview(paras[i - 1], date, paras[i + 1] + " " + paras[i + 4])
            reviews.append(rv)
        i += 1
    return reviews


def create_review_file(dir_path: pathlib.Path, review: RawReview):
    if not dir_path.is_dir():
        raise ValueError("dir_path must contain a path to a directory")

    file_name = review.date_time + "_" + review.author + ".txt"
    print(file_name)
    path = dir_path / file_name
    f = open(path, "a")
    f.write(review.text)
