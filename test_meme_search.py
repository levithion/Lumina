from meme_retrieval import _text_score


def test_exact_caption_scores_highest():
    payload = {"normalized_text": "me after deploying on friday", "template": "", "tags": []}
    assert _text_score("deploying on friday", payload) == 1.0


def test_term_overlap_is_partial_match():
    payload = {"normalized_text": "meeting could be an email", "template": "", "tags": []}
    assert 0 < _text_score("email work", payload) < 1
