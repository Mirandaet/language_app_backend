import pykakasi

def japanese_to_romanji(text):
    # Initialize the Kakasi object
    kakasi = pykakasi.kakasi()
    kakasi.setMode('H', 'a')  # Hiragana to alphabet
    kakasi.setMode('K', 'a')  # Katakana to alphabet
    kakasi.setMode('J', 'a')  # Japanese(Kanji) to alphabet
    kakasi.setMode('r', 'Hepburn')  # Use Hepburn Romanization

    # Create the converter
    converter = kakasi.getConverter()

    # Convert and return the text
    return converter.do(text)

