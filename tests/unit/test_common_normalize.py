from src.common.normalize import clean_legal_text, prettify_doc_title, normalize_question, extract_legal_signals, tokenize_vietnamese

def test_clean_legal_text():
    raw = "  Thông tư  12/2020/TT-BCA \n\n  quy định về...  "
    cleaned = clean_legal_text(raw)
    assert cleaned == "Thông tư 12/2020/TT-BCA quy định về..."

def test_prettify_doc_title():
    slug = "Nghi-dinh-44-2023-ND-CP-giam-thue-gtgt-570123"
    pretty = prettify_doc_title(slug)
    assert "Nghị định 44/2023/NĐ-CP" in pretty

def test_extract_legal_signals():
    text = "Theo quy định tại Điều 10, Khoản 2 Thông tư 58/2020/TT-BCA năm 2020..."
    signals = extract_legal_signals(text)
    assert "58/2020/TT-BCA" in signals["doc_numbers"]
    assert "10" in signals["articles"]
    assert "2" in signals["clauses"]
    assert "2020" in signals["years"]

def test_tokenize_vietnamese():
    text = "Đăng ký xe máy tại cơ quan công an"
    tok = tokenize_vietnamese(text)
    assert len(tok) > 0
