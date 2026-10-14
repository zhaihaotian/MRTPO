Checker copies of IFEval (google-research/instruction_following_eval, Apache-2.0) and IFBench (allenai/IFBench,
Apache-2.0) from HIR's evaluation folder. Changes, none of which changes the result of a response that the original
code can score: IFBench checkers return False instead of raising IndexError / ZeroDivisionError on responses with no
words, no trigrams or punctuation-only sentences (EmojiSentenceChecker, the trigram-overlap checker, the word-count and
sentence-link checkers).
