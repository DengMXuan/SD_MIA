import numpy as np
import pytest

from prepare import clean_text, match_lengths, hash_ids


def test_wikitext_restoration_keeps_real_at_and_numbers():
    assert clean_text('Contact a@b.org or @name. A honey @-@ rich cake costs 1 @,@ 234 @.@ 5 .') == (
        'Contact a@b.org or @name. A honey-rich cake costs 1,234.5.')


def test_quotes_contractions_and_footnote_cleanup():
    text='" Dance " is Stan \'s song ( 2014 ) . It did n\'t fail [ 12 ].\n^ Source citation'
    assert clean_text(text)=='"Dance" is Stan\'s song (2014). It didn\'t fail.'


def test_preserves_math_units_and_real_brackets():
    assert clean_text('Let x = 3 + 4 and A = [x, y]. Water is H2O; 10% remain.') == (
        'Let x = 3 + 4 and A = [x, y]. Water is H2O; 10% remain.')


def test_html_unicode_and_editorial_text():
    assert clean_text('A\ufeff &amp; B\xa0 [ edit ]\nReferences\nBibliographic text.')=='A & B'


def test_quoted_letters_are_not_contractions():
    text="It makes special 'M' blocks appear. He said, 'Go write it.' Once done, ' Hey ' was heard."
    assert clean_text(text)==text


def test_empty_quote_pairs_do_not_shift_on_second_pass():
    text='He sang " Song "" set off fireworks . " A writer said " good " .'
    assert clean_text(clean_text(text))==clean_text(text)


def test_translation_templates_and_stubs_removed_but_body_preserved():
    text=('You can help expand this article with text translated from Polish.\n'
          'Machine translation, like DeepL or Google Translate, is a useful starting point.\n'
          'For more guidance, see Wikipedia:Translation .\n'
          'Nida Canal connects two lakes.\n'
          'This article about a canal is a stub. You can help Wikipedia.')
    assert clean_text(text)=='Nida Canal connects two lakes.'


@pytest.mark.parametrize('text', ['a @-@ b , c .', 'A \'s " hello " .', 'x [ 1 ] ( a ) .'])
def test_idempotent(text):
    assert clean_text(clean_text(text))==clean_text(text)


def test_matching_exact_histogram_and_capacities():
    capacities=[512,240,512,200,400,512]
    targets=[200,350,180,512,300,220]
    matched=match_lengths(capacities,targets,1919)
    assert sorted(matched)==sorted(targets)
    assert np.all(np.array(matched)<=capacities)
    assert matched==match_lengths(capacities,targets,1919)


def test_infeasible_match_fails_instead_of_padding_or_dropping():
    with pytest.raises(ValueError,match='cannot supply'):
        match_lengths([128,128],[128,256],1919)


def test_hash_contract_little_endian_u32():
    import hashlib
    assert hash_ids([0,1,300])==hashlib.sha256(b'\0\0\0\0\1\0\0\0,\1\0\0').hexdigest()
