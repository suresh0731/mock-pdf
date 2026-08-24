from app.services.pii import field_labels


def test_account_name_labels_cover_en_and_id():
    assert "a/c name" in field_labels.ACCOUNT_NAME_LABELS
    assert "nama rekening" in field_labels.ACCOUNT_NAME_LABELS


def test_account_number_labels_cover_en_and_id():
    assert "account no" in field_labels.ACCOUNT_NUMBER_LABELS
    assert "no rekening" in field_labels.ACCOUNT_NUMBER_LABELS
    assert "nomor rekening" in field_labels.ACCOUNT_NUMBER_LABELS


def test_debit_credit_section_labels_are_distinct():
    assert not set(field_labels.DEBIT_SECTION_LABELS) & set(field_labels.CREDIT_SECTION_LABELS)


def test_prose_marker_includes_a_n_and_atas_nama():
    assert "a/n" in field_labels.PROSE_NAME_MARKERS
    assert "atas nama" in field_labels.PROSE_NAME_MARKERS


def test_job_title_stopwords_exclude_common_titles():
    assert "head" in field_labels.JOB_TITLE_STOPWORDS
    assert "department" in field_labels.JOB_TITLE_STOPWORDS


def test_org_prefix_stopwords_exclude_pt_and_bank():
    assert "pt" in field_labels.ORG_PREFIX_STOPWORDS
    assert "bank" in field_labels.ORG_PREFIX_STOPWORDS


def test_field_role_literal_has_five_roles():
    import typing

    roles = typing.get_args(field_labels.FieldRole)
    assert set(roles) == {
        "debit_account_name",
        "credit_account_name",
        "counterparty_org",
        "bank_name",
        "signatory_person",
    }
