def merge_texts(texts: list[str], backbone: str) -> str:
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    longest = max(texts, key=len)
    if len(longest) > len(backbone) * 1.1:
        return longest
    return backbone
