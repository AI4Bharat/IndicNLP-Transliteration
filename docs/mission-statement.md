# Mission
**Open-source Input Tools ecosystem to aid in digitalization & content creation for Indian languages including under-represented languages**
<br>

## Societal Impact

Languages are important in preserving the Cultural Identity of the people. When a language disappears, traditional knowledge and cultural heritage are also lost along with it. Throughout history, multilingualism has been threatened by various socio-political conditions. In order to preserve multilingualism and multiculturalism UNO has passed a resolution where all the member states were asked “to promote the preservation and protection of all languages used by peoples of the world”. Development in Technology and evolution of the digital world has posed different sets of challenges to Multilingualism.

India has rich and diverse languages, and cultures developed over hundreds & thousands of years. Any two people selected at random from India will have different native languages in 91.4% of cases (according to Greenberg’s Diversity Index). It is more of a responsibility to encourage the digital existence and development of all languages, including the endangered ones, thereby aiding people and community to uphold the right granted to them by Constitution of India in Article 29, which states “Any section of the citizens residing in the territory of India or any part thereof having a distinct language, script or culture of its own shall have the right to conserve the same”

Most languages in India do not have digital presence due to an underdeveloped ecosystem.  One of the major bottlenecks in content creation and language adoption, is difficulty to input text in several native Indian languages. Lack of stable input tools in underserved languages is one of the biggest roadblocks to creating digital content and datasets in these languages.
The main goal of this project is to create open source input tools for under-represented languages in India, which would not only solve few of the problems for under-represented languages but also motivate technocrats from these under-represented communities to extend more solutions for their languages. Thereby creating a profound digital presence and accelerating the development.


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
