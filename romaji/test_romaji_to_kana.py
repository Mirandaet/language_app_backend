import MeCab

def parse_text(text):
    tagger = MeCab.Tagger()
    parsed = tagger.parse(text)
    return parsed

# Example usage
japanese_text = "私の名前は中野です"
result = parse_text(japanese_text)
print(result)
