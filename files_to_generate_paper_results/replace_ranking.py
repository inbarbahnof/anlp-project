import json

with open("../mmlu_questions_by_subject.json", "r", encoding="utf-8") as f:
    MMLU_QUESTIONS = json.load(f)

with open("mmlu_scores_dove.json", "r", encoding="utf-8") as f:
    MMLU_RANKINGS = json.load(f)

TARGET_MODEL = "Llama-3.1-8B-Instruct"


# Dummy placeholder function – replace with your actual logic
def get_ranking_from_question(question):
    for subject, questions in MMLU_QUESTIONS.items():
        for index_str, q_text in questions.items():
            if q_text.strip() == question.strip():
                # Found the matching question
                index = index_str  # this is a string
                if index in MMLU_RANKINGS:
                    return MMLU_RANKINGS[index]  # return the float score
                else:
                    return -1  # ranking not found for that index
    return -1  # question not found


def update_rankings(node):
    # If "input" and "ranking" keys exist, and the input starts with the question pattern
    if isinstance(node, dict):
        if "input" in node and isinstance(node["input"], str) and node["input"].startswith(
                "Question:"):
            full_text = node["input"].split("Question:")[1].strip()
            question = full_text.split("\n\n")[0].strip()
            score = get_ranking_from_question(question)
            node["ranking"] = [[TARGET_MODEL, score]]

        # Recurse into subtrees if they exist
        if "subtrees" in node and isinstance(node["subtrees"], list):
            for subtree in node["subtrees"]:
                update_rankings(subtree)


def main():
    # Load the JSON from file
    with open("../MMLU.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Update rankings
    update_rankings(data)

    # Save the updated JSON
    with open("evaltree_with_dove_result.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
