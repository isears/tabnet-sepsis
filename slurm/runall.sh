#!/bin/bash
# Do everything

for i in 6 9 12 15 18 21 24
do
    # Training
    python src/tabsep/modeling/lrRunner.py train cache/sparse_labeled_$i.pkl
    python src/tabsep/modeling/tabnetRunner.py train cache/sparse_labeled_$i.pkl
    python src/tabsep/modeling/skorchTSTRunner.py train cache/sparse_labeled_$i.pkl

    # CV
    python src/tabsep/modeling/lrRunner.py cv cache/sparse_labeled_$i.pkl
    python src/tabsep/modeling/tabnetRunner.py cv cache/sparse_labeled_$i.pkl
    python src/tabsep/modeling/skorchTSTRunner.py cv cache/sparse_labeled_$i.pkl

    # Attribution
    python src/tabsep/modeling/tabnetRunner.py captum cache/sparse_labeled_$i.pkl
    python src/tabsep/modeling/skorchTSTRunner.py captum cache/sparse_labeled_$i.pkl
done