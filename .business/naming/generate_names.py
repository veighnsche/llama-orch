#!/usr/bin/env python3
"""
Name Generator for Product Naming
Generates pronounceable, brandable names using weighted phonetic clusters.
"""

import random
from collections import defaultdict

# Cluster inventory
PREFIXES = [
    "gl","gr","tr","fl","fr","pl","pr","cr","cl","dr","sk","sl","sn","sp",
    "bl","br","st","str","thr","tw","dw","ch","sh","th","wh",
    "v","z","x","k","n","m","r","t","p","b","f","g","d","s","l"
]
VOWELS = ["a","e","i","o","u","y"]
SUFFIXES = [
    "x","z","n","r","t","k","g","v","m","p","q","l","f","s","th",
    "sh","ch","nk","nd","nt","rn","rm","rt","st","sp","sk","ff"
]

# Map first-letter -> allowed onsets that start with that letter (plus bare letter)
onsets_by_letter = defaultdict(list)
for onset in PREFIXES:
    onsets_by_letter[onset[0]].append(onset)
for letter in set(onsets_by_letter.keys()).union(set("stvzfgpncklrbmdaehioquyxw")):
    onsets_by_letter[letter].append(letter)

# Default weights (edit these to taste)
DEFAULT_WEIGHTS = {
    # 🥇 hot & modern
    **{c: 10 for c in "stvz"},
    # 🥈 infra & dev-friendly
    **{c:  6 for c in "fgpncklr"},
    # 🥉 secondary
    **{c:  3 for c in "bmda"},
    # 🚫 caution (almost none)
    **{c:0.5 for c in "ehioquy"},
    # rare edgy starters
    "x": 1, "w": 1,
}

def weighted_choice(weights: dict) -> str:
    """Choose a letter based on weighted probabilities."""
    letters, probs = zip(*weights.items())
    total = sum(probs)
    r = random.random() * total
    acc = 0.0
    for c, p in zip(letters, probs):
        acc += p
        if r <= acc:
            return c
    return letters[-1]

def generate_word(weights=DEFAULT_WEIGHTS) -> str:
    """Generate a single pronounceable word."""
    first = weighted_choice(weights)
    onset = random.choice(onsets_by_letter[first])
    vowel = random.choice(VOWELS)
    suffix = random.choice(SUFFIXES)
    return (onset + vowel + suffix).lower()

def generate_word_list(count=1000, weights=DEFAULT_WEIGHTS, unique=True):
    """Generate a list of words."""
    words = set() if unique else []
    attempts = 0
    max_attempts = count * 10  # Prevent infinite loops
    
    while len(words) < count and attempts < max_attempts:
        w = generate_word(weights)
        if unique:
            words.add(w)
        else:
            words.append(w)
        attempts += 1
    
    return sorted(words) if unique else words

def main():
    """Generate 1000 unique names and save to file."""
    print("🐝 Generating 1000 unique product names...")
    
    # Generate names
    names = generate_word_list(count=1000, unique=True)
    
    # Write to file
    output_file = "GENERATED_NAMES.txt"
    with open(output_file, "w") as f:
        f.write("# Generated Product Names\n")
        f.write("# Total: {} unique names\n".format(len(names)))
        f.write("# Generated with weighted phonetic clusters\n")
        f.write("# Date: 2025-10-09\n\n")
        
        for i, name in enumerate(names, 1):
            f.write(f"{name}\n")
    
    print(f"✅ Generated {len(names)} unique names")
    print(f"📄 Saved to: {output_file}")
    print(f"\n🎯 First 20 samples:")
    for name in names[:20]:
        print(f"   - {name}")

if __name__ == "__main__":
    main()
