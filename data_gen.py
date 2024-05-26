import random
from tqdm import tqdm
import multiprocessing
import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="stack_depth",
        choices=["stack_depth", "count"],
        help="Type of data to generate, either based on stack depth or count",
    )
    parser.add_argument(
        "--only_unbalanced",
        action="store_true",
        help="Generate only unbalanced data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate per stack depth or per count",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=512,
        help="Length of the sequence to generate",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=0,
        help="Minimum count of brackets to generate",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=0,
        help="Maximum count of brackets to generate",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="Data",
        help="Path to save the generated data",
    )
    args = parser.parse_args()

    if args.only_unbalanced:
        assert (
            args.type == "count"
        ), "Only unbalanced data can be generated for count type"

    if args.type == "stack_depth":
        args.save_name = (
            f"{args.type}_seq-len-{args.length}_per-stack-depth-{args.num_samples}"
        )
    else:
        args.save_name = f"{args.type}_min-{args.min_count}_max-{args.max_count}_per-count-{args.num_samples}"
        if args.only_unbalanced:
            args.save_name += "_only-unbalanced_testing"
    return args


def generate_balanced_brackets(n):
    sequence = ""
    stack = []

    for i in range(n):
        open_or_close = random.choice([0, 1]) and stack
        left = n - i
        if left < len(stack) or (not open_or_close and len(stack) == left):
            open_or_close = 1
        if open_or_close:
            sequence += ")"
            stack.pop()
        else:
            sequence += "("
            stack.append("(")

    return sequence


def generate_balanced_brackets_with_stack_depth(n, stack_depth):
    sequence = ""
    stack = []

    # for i in range(min_depth):
    #     sequence += '('
    #     stack.append('(')
    start = random.randint(0, n - 2 * stack_depth)
    # print(start)

    for i in range(n):
        if i > start and i < start + stack_depth and stack_depth > len(stack):
            sequence += "("
            stack.append("(")
            continue
        open_or_close = random.choice([0, 1]) and stack
        left = n - i
        # print(left, n, i, min_depth)
        if (
            left < len(stack)
            or (not open_or_close and len(stack) == left)
            or len(stack) == stack_depth
        ):
            open_or_close = 1
        if open_or_close:
            sequence += ")"
            stack.pop()
        else:
            sequence += "("
            stack.append("(")

    return sequence


def is_balanced(s):
    stack = []
    max_depth = 0
    count = 0

    for bracket in s:
        if bracket == ")":
            if stack:
                stack.pop()
            else:
                stack.append(bracket)
                break
        else:
            stack.append(bracket)
            max_depth = max(max_depth, len(stack))

    for bracket in s:
        if bracket == "(":
            count += 1
        else:
            count -= 1

    if stack:
        return -1 * max_depth, count
    return max_depth, count


def generate_unbalanced_with_stack_depth(n, stack_depth):
    seq = generate_balanced_brackets_with_stack_depth(n, stack_depth)
    bal = is_balanced(seq)
    seq2 = seq
    while bal[0] != -stack_depth:
        seq2 = seq
        index = random.randint(0, n - 1)
        if seq[index] == "(":
            seq2 = seq[:index] + ")" + seq[index + 1 :]
        else:
            seq2 = seq[:index] + "(" + seq[index + 1 :]
        bal = is_balanced(seq2)
    return seq2


def generate_unbalanced_with_count(length, count, num_samples):
    gen_set = set()
    flag = 0
    open_no = 0
    close_no = 0
    for i in range(num_samples):
        if random.random() > 0.5:
            flag = 1
        if flag == 1:
            open_no = length / 2 + (count) / 2
            close_no = length / 2 - (count) / 2
        else:
            open_no = length / 2 - (count) / 2
            close_no = length / 2 + (count) / 2

        sequence = "(" * int(open_no) + ")" * int(close_no)
        # randomly permute

        sequence = "".join(random.sample(sequence, len(sequence)))
        while is_balanced(sequence)[0] > 0:
            sequence = "".join(random.sample(sequence, len(sequence)))
        if sequence not in gen_set:
            gen_set.add(sequence)

    return list(gen_set)


def balanced_df_stack_depths(gen, length, num_samples):
    # Generating according to stack depth
    balanced_df = pd.DataFrame(columns=["sequence", "stack_depth", "count"])
    seqs = [0 for i in range(400)]
    generated_seqs = set()
    print(f"Generating sequences in the following stack depths: {gen}")
    for j in tqdm(gen):
        while True:

            sequence = generate_balanced_brackets_with_stack_depth(length, j)
            balanced, count = is_balanced(sequence)

            if (
                seqs[balanced] == num_samples
                or sequence in generated_seqs
                or balanced not in gen
            ):
                continue

            seqs[balanced] += 1

            balanced_df.loc[len(balanced_df)] = [sequence, balanced, count]
            generated_seqs.add(sequence)

            if seqs[j] >= num_samples:
                break
        # print(seqs)
    return balanced_df


def unbalanced_df_stack_depth(gen, length, num_samples):
    unbalanced_data = []
    for index, i in tqdm(enumerate(gen), total=len(gen)):
        while len(unbalanced_data) < num_samples * (index + 1):
            sequence = generate_unbalanced_with_stack_depth(length, i)
            unbalanced_data.append(sequence)

        # print(len(unbalanced_data))

    print(len(unbalanced_data))
    unbalanced_stack_depth = [is_balanced(sequence) for sequence in unbalanced_data]

    unbalanced_count = [x[1] for x in unbalanced_stack_depth]
    unbalanced_stack_depth = [x[0] for x in unbalanced_stack_depth]

    unbalanced_df = pd.DataFrame(columns=["sequence", "stack_depth", "count"])
    unbalanced_df["sequence"] = unbalanced_data
    unbalanced_df["stack_depth"] = unbalanced_stack_depth
    unbalanced_df["count"] = unbalanced_count

    return unbalanced_df


def balanced_df_count(length, num_samples):
    balanced_df = pd.DataFrame(columns=["sequence", "stack_depth", "count"])
    for _ in range(num_samples):
        sequence = generate_balanced_brackets(length)
        balanced, count = is_balanced(sequence)
        balanced_df.loc[len(balanced_df)] = [sequence, balanced, count]
    return balanced_df


def unbalanced_df_count(length, num_samples, min_count, max_count):
    unbalanced_data = []
    for count in tqdm(range(min_count, max_count + 1, 2)):
        print(f"generating unbalanced data for count: {count}")
        unbalanced_data += generate_unbalanced_with_count(length, count, num_samples)

    unbalanced_stack_depth = [is_balanced(sequence) for sequence in unbalanced_data]
    unbalanced_count = [x[1] for x in unbalanced_stack_depth]
    unbalanced_stack_depth = [x[0] for x in unbalanced_stack_depth]

    unbalanced_df = pd.DataFrame(columns=["sequence", "stack_depth", "count"])
    unbalanced_df["sequence"] = unbalanced_data
    unbalanced_df["stack_depth"] = unbalanced_stack_depth
    unbalanced_df["count"] = unbalanced_count

    return unbalanced_df


def save_csv(bal_df, unbal_df, save_dir, save_name):
    # concat balanced and unbalanced dataframes and shuffle
    if bal_df is None:
        df = unbal_df
    else:
        df = pd.concat([bal_df, unbal_df], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)

    # save the dataframe
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(os.path.join(save_dir, save_name + ".csv"), index=False)


if __name__ == "__main__":
    gen = [i for i in range(2, 10)] + [i for i in range(10, 101, 10)]

    config = parse_args()

    # assert max and min count are even
    assert config.max_count % 2 == 0
    assert config.min_count % 2 == 0

    if config.type == "stack_depth":
        bal_df = balanced_df_stack_depths(gen, config.length, config.num_samples)
        print(bal_df.shape)
        unbal_df = unbalanced_df_stack_depth(gen, config.length, config.num_samples)
        print(unbal_df.shape)
    else:
        if not config.only_unbalanced:
            bal_df = balanced_df_count(
                config.length,
                config.num_samples * ((config.max_count - config.min_count) // 2 + 1),
            )
            print(bal_df.shape)
        else:
            bal_df = None
        unbal_df = unbalanced_df_count(
            config.length, config.num_samples, config.min_count, config.max_count
        )
        print(unbal_df.shape)

    save_csv(bal_df, unbal_df, config.save_dir, config.save_name)
