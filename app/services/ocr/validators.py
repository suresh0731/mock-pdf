import re

import phonenumbers


def validate_ssn(value: str) -> bool:
    cleaned = re.sub(r"\D", "", value)
    if len(cleaned) != 9:
        return False
    area = int(cleaned[:3])
    return area not in (0, 666) and area < 900


def validate_nric(value: str) -> bool:
    return bool(re.fullmatch(r"[STFGM]\d{7}[A-Z]", value.strip()))


def validate_mykad(value: str) -> bool:
    return bool(re.fullmatch(r"\d{6}-\d{2}-\d{4}", value.strip()))


def validate_hkid(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,2}\d{6}\([0-9A]\)", value.strip()))


def validate_nik(value: str) -> bool:
    return bool(re.fullmatch(r"\d{16}", re.sub(r"\D", "", value)))


def validate_email(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value.strip()))


def validate_phone(value: str, locale: str | None = None) -> bool:
    try:
        region = {"sg": "SG", "my": "MY", "hk": "HK", "id": "ID", "us": "US"}.get(locale or "", "US")
        parsed = phonenumbers.parse(value, region)
        return phonenumbers.is_valid_number(parsed)
    except Exception:
        digits = re.sub(r"\D", "", value)
        return 8 <= len(digits) <= 15


def validate_entity(entity_type: str, value: str, locale: str | None = None) -> bool:
    et = entity_type.upper()
    if et in ("US_SSN", "SSN"):
        return validate_ssn(value)
    if et == "NRIC":
        return validate_nric(value)
    if et in ("MYKAD", "MY_KAD"):
        return validate_mykad(value)
    if et == "HKID":
        return validate_hkid(value)
    if et == "NIK":
        return validate_nik(value)
    if et == "EMAIL_ADDRESS":
        return validate_email(value)
    if et == "PHONE_NUMBER":
        return validate_phone(value, locale)
    return True
