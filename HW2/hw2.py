# TCSS455: Machine Learning
# Homework 2
# Name: Martine De Cock, Dilnoza Saidova
# Description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd
import operator
from collections import Counter


class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children.
    # The attribute is either the attribute being split on,
    # or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level=0):
        if self.children == {}:  # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)

    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}:  # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree


def id3(examples, target, attributes):
    examples_list = examples.loc[:, target].tolist()

    if examples_list.count(examples_list[0]) == len(examples_list):
        return DecisionNode(examples_list[0])

    elif len(attributes) == 0:
        example_cnt = {}
        for example in examples_list:
            if example not in example_cnt.keys():
                example_cnt[example] = 1
            example_cnt[example] += 1
        sorted_cnt = sorted(example_cnt.items(),
                            key=operator.itemgetter(1),
                            reverse=True)
        return DecisionNode(sorted_cnt[0][0])

    else:
        final_gain = 0.0
        attr_index = 0

        for num in range(len(attributes)):
            entropy = getEntropy(examples, target)
            attr_cnt = Counter(examples.loc[:, attributes[num]])
            new_entropy = 0.0

            for key in attr_cnt:
                temp = examples.loc[examples[attributes[num]] == key]
                del temp[attributes[num]]
                mod_examples = temp
                new_entropy += (attr_cnt[key] / sum(attr_cnt.values())) \
                               * getEntropy(mod_examples, target)
            new_gain = (entropy - new_entropy)

            if new_gain > final_gain:
                final_gain = new_gain
                attr_index = num

        top_attr = attributes[attr_index]
        tree = DecisionNode(top_attr)
        top_attr_cnt = Counter(examples.loc[:, top_attr])
        values = []

        for key in top_attr_cnt:
            if key not in values:
                values.append(key)

        for value in values:
            temp = examples.loc[examples[top_attr] == value]
            del temp[top_attr]
            new_ex = temp

            new_attr = attributes[:]
            new_attr.remove(top_attr)
            subtree = id3(new_ex, target, new_attr)
            tree.children[value] = subtree
    return tree


def getEntropy(examples, target):
    cnt = Counter(examples.loc[:, target])
    sum = 0
    for num in cnt:
        sum += -1.0 * (cnt[num] / len(examples)) \
               * math.log(cnt[num] / len(examples), 2)
    return sum


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train, target, attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0, len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i, target]):
        correct += 1
print("\nThe accuracy is: ", correct / len(test))
