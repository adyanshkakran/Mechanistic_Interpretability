import random
import pandas as pd
import csv
from tqdm import tqdm

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

def generate_bracket_sequence(n):
    brackets = ["(", ")"]
    sequence = ""

    for i in range(n):
        sequence += random.choice(brackets)

    if is_balanced(sequence)[0] < 0:
        return sequence
    else:
        return generate_bracket_sequence(n)

def generate_unbalanced_brackets(n):
    seq = generate_bracket_sequence(n)
    bal = is_balanced(seq)
    seq2 = seq
    while bal[0] > 0:
        print('yes')
        seq2 = seq
        index = random.randint(0, n-1)
        if seq[index] == '(':
            seq2 = seq[:index] + ')' + seq[index+1:]
        else:
            seq2 = seq[:index] + '(' + seq[index+1:]
        bal = is_balanced(seq2)
    return seq2

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

if __name__=="__main__":
    balanced_brackets = []
    unbalanced_brackets = []
    for i in tqdm(range(200)):
        # select a random length for the bracket sequence between 2 and 50
        n = random.randint(52, 100)
        while n % 2 != 0:
            n = random.randint(52, 100) # make sure the length is even
        
        # generate a balanced bracket sequence
        balanced = generate_balanced_brackets(n)
        # generate an unbalanced bracket sequence
        unbalanced = generate_unbalanced_brackets(n)
        
        # add the balanced and unbalanced sequences to the dictionary
        balanced_brackets.append({
            'sequence': balanced,
            'stack_depth': is_balanced(balanced)[0],
            'count': is_balanced(balanced)[1],
        })

        unbalanced_brackets.append({
            'sequence': unbalanced,
            'stack_depth': is_balanced(unbalanced)[0],
            'count': is_balanced(unbalanced)[1],
        })
    
    # export into csv file
    df = pd.DataFrame(balanced_brackets)
    df.to_csv('Data/new_new_balanced_brackets.csv', index=False)
    
    df = pd.DataFrame(unbalanced_brackets)
    df.to_csv('Data/new_new_unbalanced_brackets.csv', index=False)
        
        