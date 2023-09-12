import re

def calculate_scores(text):
    sentences = re.split(r'[.!?]', text)
    word_count = 0
    syllable_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)
        for word in words:
            syllable_count += count_syllables(word)

    if word_count == 0:
        return 0, 0
    else:
        fk_score = 206.835 - 1.015 * (word_count / len(sentences)) - 84.6 * (syllable_count / word_count)
        fk_grade_level = 0.39 * (word_count / len(sentences)) + 11.8 * (syllable_count / word_count) - 15.59
        return fk_score, fk_grade_level

def count_syllables(word):
    word = word.lower()
    if len(word) <= 3:
        return 1
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count = 1
    return count

if __name__ == "__main__":
    text = input("Enter the text: ")
    fk_score, fk_grade_level = calculate_scores(text)
    print(f"Flesch Reading Ease Score: {fk_score:.2f}")
    print(f"Flesch-Kincaid Grade Level: {fk_grade_level:.2f}")
