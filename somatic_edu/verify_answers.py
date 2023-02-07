ANSWER_PATH = 'labels.txt'
TRUE_ANSWER_PATH = 'true_labels.txt'


def load_answers(path) -> dict:
    answers = dict()
    with open(path) as file:
        for line in file.readlines():
            example_name, crosses, contours = line.split()
            answers[example_name] = crosses, contours
    return answers


def main():
    answs = load_answers(ANSWER_PATH)
    true_answs = load_answers(TRUE_ANSWER_PATH)

    wrong_count = len(true_answs)
    for example_name, value in answs.items():
        if true_answs[example_name] == value:
            wrong_count -= 1
            print(f"{example_name}... Ok")
        else:
            print(f"{example_name}... Wrong")
    correct = len(true_answs) - wrong_count
    print(f"Guessed: {correct} of {len(true_answs)} examples")


if __name__ == '__main__':
    main()
