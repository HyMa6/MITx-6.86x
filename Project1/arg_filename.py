def foo (stopwords=None):
    with open('stopwords.txt') as s:
        lines = s.readlines()
        print(lines)

foo(stopwords=None)