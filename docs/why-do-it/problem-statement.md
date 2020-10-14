# Problem Statement

Due to prominent use of English in the technological ecosystems, language interfaces built for English are stable and compatible with large number devices and applications, and continuously developing.
Also given the current comfort of the people in using the tools built around this ecosystem,
it would be meaningful to build tools that comply with an existing ecosystem rather than building entirely different tools and systems. Because of this people could adapt to these tools with minimal effort making it easier and productive and thereby impact many lives.
Given the above considerations, building a Transliteration engine that supports english to Native Languages is a good first step in ensuring digital presence of Indian languages, more specifically under-represented languages and wider adoption by native speakers.


## Challenges

- Each language has different variants of phonemes and pronunciations based on the origins and historical evolution of the language.
- Representing one language using the character of a language that has significant linguistic difference, would degenerate many of the rich phonemes and orthography.
- A rule based one-one mapping of characters for transliteration is not very successful because of the difference in usage by people, users tend to adapt a certain representation that is influenced by trend and the native language of the user. Like using “zha” for deep ‘la’ sound “ழ” by tamil users.
- Also the latin characters used to construct phonemes of the native words, differ from word to word.

## Approach

- Create a learning based approach for building models that could understand the latin character usage pattern of native language specific users
- Understand the orthography of the native language
- Build user adaptive model that could accomodate for variation among users.
