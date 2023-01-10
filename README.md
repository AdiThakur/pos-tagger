## Part-of-Speech Tagger

This `python` script uses Hidden Markov Models and the `Viterbi` algorithm to perform [part-of-speech](https://en.wikipedia.org/wiki/Part-of-speech_tagging) (`POS`) tagging on a given file. It can be invoked using the following command

```
python3 tagger.py -d <training files> -t <test file> -o <output file>
```

---

#### Input Format

`<training files>` are plain text files that have already been tagged. Such files consist of many lines, each of which is formatted as `word : tag`.

`<test file>` is a plain text file that you wish to tag. Each line of this file must contain exactly one word/punctuation character (whitespace should not be included). For example, the text `"Hello, world?"` would be formatted as
```
"
Hello
,
world
?
"
```

#### Output Format

The output format is exactly the same as the specification for the `<training files>`.