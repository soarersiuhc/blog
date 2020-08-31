---
layout: post
title: "Clean code reading notes"
page_name: Clean-code-reading-notes
date: 2020-08-30
categories: 
---

# Principles

### Be expressive
- Chapter 2
  - Expressive names
    - Pronounceable names
    - Searchable names
    - Use solution domain/problem domain names
    - Naming conventions
      - Class name: Noun
      - Method names:
        - Verb
        - Instead of overloading constructor, consider factory method with different names
      - Pick one word per concept (e.g. use only get, instead of using fetch/get/retrieve at different places)
  - Avoid Disinformation (e.g. accountList should be a list)

### No surprise
- Chapter 2
  - Avoid Disinformation (e.g. accountList should be a list)
  - Avoid mental mapping (e.g. don't use your-latest-cool-variable-name instead of i or j for for loops - one unnecessary layer of mental translation)
---

# Chapter 1
- Cost of bad code
- What is clean code?
  - Bjarne Stroustrup's quote
    - > I like my code to be elegant and efficient. The logic should be straightforward to make it hard for bugs to hide, the dependencies minimal to ease maintenance, error handling complete according to an articulated strategy, and performance close to optimal so as not to tempt people to make the code messy with unprincipled optimizations. Clean code does one thing well.
  - Grady Booch
    - > Clean code is simple and direct. Clean code reads like well-written prose. Clean code never obscures the designer’s intent but rather is full of crisp abstractions and straightforward lines of control.
- Boy Scout Rule: Leave the campground cleaner than you found it.
