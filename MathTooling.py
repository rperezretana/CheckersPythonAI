import re


def base4_to_decimal(base4_num_str):
    """Convert a base 4 number (as a string) to a decimal number."""
    decimal_num = 0
    for digit in base4_num_str:
        decimal_num = decimal_num * 4 + int(digit)
    return decimal_num

def decimal_to_base72(decimal_num):
    """Convert a decimal number to a base 72 string."""
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()"
    if decimal_num == 0:
        return "0"
    base72_str = ""
    while decimal_num > 0:
        decimal_num, rem = divmod(decimal_num, 72)
        base72_str = alphabet[rem] + base72_str
    return base72_str


def base4_to_base72(base4_num_str):
    """Convert a base 4 number (as a string) to a base 72 string."""
    decimal_num = base4_to_decimal(base4_num_str)
    return decimal_to_base72(decimal_num)


def clean_string(input_string):
    # Use regex to keep only numbers and negative signs
    cleaned_string = re.sub(r'[^\d-]', '', input_string)
    return cleaned_string


def transform_dict_keys_base4_to_base72(input_dict):
    """Transform dictionary keys from base 4 to base 72."""
    transformed_dict = {}
    for key, value in input_dict.items():
        key = clean_string(key)
        key = transform_key_to_base_4(key)
        new_key = base4_to_base72(key)
        transformed_dict[new_key] = value
    return transformed_dict


def transform_key_to_base_4(key):
    key = key.replace('-2', '3').replace('-1', '4')
    return key

def transform_key_to_base72(key):
    key = clean_string(key)
    # transform it to base4 from regular key
    key = transform_key_to_base_4(key)
    # transform it to base72
    return base4_to_base72(key)
