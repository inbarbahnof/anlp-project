# ============================================================================
# answer_mapper.py
"""Answer mapping functionality for multiple choice questions."""

import logging
from typing import Optional, Set, Dict, Union, List


class AnswerMapper:
    """Maps multiple choice answer strings to valid positions (1-4)."""

    # Mapping configurations for different answer formats
    POSITION_MAPPINGS = {
        "greek": "αβγδεζηθικ",
        "keyboard": "!@#$%^₪*)(",
        "capitals": "ABCDEFGHIJ",
        "lowercase": "abcdefghij",
        "numbers": [str(i + 1) for i in range(10)],
        "roman": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
    }

    VALID_POSITIONS: Set[int] = {1, 2, 3, 4}

    @classmethod
    def get_position(cls, answer: str) -> Optional[int]:
        """
        Map an answer to a valid multiple choice position (1-4).

        Args:
            answer: The answer string to map

        Returns:
            Position (1-4) if valid, None otherwise
        """
        logging.debug(f"AnswerMapper: Processing '{answer}' (Type: {type(answer)})")

        if not isinstance(answer, str):
            logging.debug(f"AnswerMapper: Input is not a string. Returning None.")
            return None

        cleaned_answer = answer.strip()
        if not cleaned_answer:
            logging.debug("AnswerMapper: Empty string after strip. Returning None.")
            return None

        # Extract potential label before separator
        prefix_to_map = cls._extract_label_prefix(cleaned_answer)

        # Clean common prefixes
        prefix_to_map = cls._clean_common_prefixes(prefix_to_map)

        # Try to map the cleaned prefix
        position = cls._map_prefix_to_position(prefix_to_map)

        logging.debug(f"AnswerMapper: Final result for '{answer}': {position}")
        return position

    @classmethod
    def _extract_label_prefix(cls, answer: str) -> str:
        """Extract the label part before common separators."""
        label_end_index = -1
        for separator in [".", ")", ":"]:
            idx = answer.find(separator)
            if idx != -1 and (label_end_index == -1 or idx < label_end_index):
                label_end_index = idx

        if label_end_index == -1:
            return answer

        # Check if separator is followed by space (common in "A. Text")
        if len(answer) > label_end_index + 1 and answer[label_end_index + 1].isspace():
            potential_label = answer[:label_end_index].strip()
            if potential_label:
                logging.debug(f"AnswerMapper: Extracted label '{potential_label}'")
                return potential_label

        return answer

    @classmethod
    def _clean_common_prefixes(cls, text: str) -> str:
        """Remove common prefixes like 'Answer: ' or 'The answer is '."""
        lower_text = text.lower()
        prefixes_to_remove = ["answer: ", "the answer is ", "the correct answer is "]

        for prefix in prefixes_to_remove:
            if lower_text.startswith(prefix):
                cleaned = text[len(prefix) :].strip()
                logging.debug(f"AnswerMapper: Cleaned prefix. New text: '{cleaned}'")
                return cleaned

        # Handle quotes around single characters
        if len(text) == 3 and text[0] in ["'", '"', "`"] and text[2] in ["'", '"', "`"]:
            return text[1]

        return text

    @classmethod
    def _map_prefix_to_position(cls, prefix: str) -> Optional[int]:
        """Map the cleaned prefix to a position using various mappings."""
        if not prefix:
            return None

        try:
            # Try each mapping type
            for mapping_name, mapping in cls.POSITION_MAPPINGS.items():
                position = cls._try_mapping(prefix, mapping, mapping_name)
                if position is not None:
                    return position

            # Direct character mapping fallback
            return cls._direct_character_mapping(prefix)

        except (AttributeError, IndexError, ValueError) as e:
            logging.debug(f"AnswerMapper: Error processing '{prefix}': {e}")
            return None

    @classmethod
    def _try_mapping(
        cls, prefix: str, mapping: Union[str, List[str]], mapping_name: str
    ) -> Optional[int]:
        """Try to map prefix using a specific mapping."""
        if isinstance(mapping, list):
            if prefix in mapping:
                position = mapping.index(prefix) + 1
                if position in cls.VALID_POSITIONS:
                    logging.debug(f"AnswerMapper: Found in {mapping_name}: {position}")
                    return position
        elif isinstance(mapping, str) and len(prefix) == 1:
            if prefix in mapping:
                position = mapping.index(prefix) + 1
                if position in cls.VALID_POSITIONS:
                    logging.debug(f"AnswerMapper: Found in {mapping_name}: {position}")
                    return position
        return None

    @classmethod
    def _direct_character_mapping(cls, prefix: str) -> Optional[int]:
        """Direct mapping for common single characters."""
        if len(prefix) != 1:
            return None

        # Capital letters A-D
        if prefix in "ABCD":
            position = ord(prefix) - ord("A") + 1
            if position in cls.VALID_POSITIONS:
                return position

        # Lowercase letters a-d
        if prefix in "abcd":
            position = ord(prefix) - ord("a") + 1
            if position in cls.VALID_POSITIONS:
                return position

        # Numbers 1-4
        if prefix in "1234":
            position = int(prefix)
            if position in cls.VALID_POSITIONS:
                return position

        return None
