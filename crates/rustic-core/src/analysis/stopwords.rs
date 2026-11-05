//! Stop word sets for the built-in language analyzers.
//!
//! Generated from Apache Lucene's analyzer resources (apache/lucene `main`,
//! `lucene/analysis/common/src/resources/org/apache/lucene/analysis/`),
//! the same files Elasticsearch ships. Verbatim copies of every source file
//! live in `testdata/stopwords/`, and the tests re-parse them with Lucene's
//! `WordlistLoader` rules and require the lists here to match exactly.
//!
//! - `snowball/<lang>_stop.txt` (Snowball format: text after `|` is a
//!   comment, whitespace-separated words) for danish, dutch, finnish,
//!   french, german, hungarian, irish, italian, norwegian, portuguese,
//!   russian, spanish and swedish.
//! - `<code>/stopwords.txt` (one word per line, `#` comment lines) for
//!   arabic, armenian, basque, bengali, brazilian, bulgarian, catalan, cjk
//!   (`CJKAnalyzer`), czech, estonian, galician, greek, hindi, indonesian,
//!   latvian, lithuanian, persian, romanian, serbian, sorani, thai and
//!   turkish.
//! - English is `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET` (33 words).
//!
//! Entries keep the source order; repeated entries appear once (Lucene loads
//! them into a set). Lucene's Galician, Latvian and Sorani analyzers load
//! their file without a comment marker, so their `#` header lines become
//! (unmatchable) set members there; they are omitted here.

/// Lucene `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET` (33 words, the ES `_english_` list).
pub const ENGLISH: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with",
];

/// Lucene `DanishAnalyzer` default set: `snowball/danish_stop.txt` (94 words).
pub const DANISH: &[&str] = &[
    "og", "i", "jeg", "det", "at", "en", "den", "til", "er", "som", "på", "de", "med",
    "han", "af", "for", "ikke", "der", "var", "mig", "sig", "men", "et", "har", "om", "vi",
    "min", "havde", "ham", "hun", "nu", "over", "da", "fra", "du", "ud", "sin", "dem", "os",
    "op", "man", "hans", "hvor", "eller", "hvad", "skal", "selv", "her", "alle", "vil",
    "blev", "kunne", "ind", "når", "være", "dog", "noget", "ville", "jo", "deres", "efter",
    "ned", "skulle", "denne", "end", "dette", "mit", "også", "under", "have", "dig",
    "anden", "hende", "mine", "alt", "meget", "sit", "sine", "vor", "mod", "disse", "hvis",
    "din", "nogle", "hos", "blive", "mange", "ad", "bliver", "hendes", "været", "thi",
    "jer", "sådan",
];

/// Lucene `DutchAnalyzer` default set: `snowball/dutch_stop.txt` (101 words).
pub const DUTCH: &[&str] = &[
    "de", "en", "van", "ik", "te", "dat", "die", "in", "een", "hij", "het", "niet", "zijn",
    "is", "was", "op", "aan", "met", "als", "voor", "had", "er", "maar", "om", "hem", "dan",
    "zou", "of", "wat", "mijn", "men", "dit", "zo", "door", "over", "ze", "zich", "bij",
    "ook", "tot", "je", "mij", "uit", "der", "daar", "haar", "naar", "heb", "hoe", "heeft",
    "hebben", "deze", "u", "want", "nog", "zal", "me", "zij", "nu", "ge", "geen", "omdat",
    "iets", "worden", "toch", "al", "waren", "veel", "meer", "doen", "toen", "moet", "ben",
    "zonder", "kan", "hun", "dus", "alles", "onder", "ja", "eens", "hier", "wie", "werd",
    "altijd", "doch", "wordt", "wezen", "kunnen", "ons", "zelf", "tegen", "na", "reeds",
    "wil", "kon", "niets", "uw", "iemand", "geweest", "andere",
];

/// Lucene `FinnishAnalyzer` default set: `snowball/finnish_stop.txt` (229 words).
pub const FINNISH: &[&str] = &[
    "olla", "olen", "olet", "on", "olemme", "olette", "ovat", "ole", "oli", "olisi",
    "olisit", "olisin", "olisimme", "olisitte", "olisivat", "olit", "olin", "olimme",
    "olitte", "olivat", "ollut", "olleet", "en", "et", "ei", "emme", "ette", "eivät",
    "minä", "minun", "minut", "minua", "minussa", "minusta", "minuun", "minulla", "minulta",
    "minulle", "sinä", "sinun", "sinut", "sinua", "sinussa", "sinusta", "sinuun", "sinulla",
    "sinulta", "sinulle", "hän", "hänen", "hänet", "häntä", "hänessä", "hänestä", "häneen",
    "hänellä", "häneltä", "hänelle", "me", "meidän", "meidät", "meitä", "meissä", "meistä",
    "meihin", "meillä", "meiltä", "meille", "te", "teidän", "teidät", "teitä", "teissä",
    "teistä", "teihin", "teillä", "teiltä", "teille", "he", "heidän", "heidät", "heitä",
    "heissä", "heistä", "heihin", "heillä", "heiltä", "heille", "tämä", "tämän", "tätä",
    "tässä", "tästä", "tähän", "tällä", "tältä", "tälle", "tänä", "täksi", "tuo", "tuon",
    "tuota", "tuossa", "tuosta", "tuohon", "tuolla", "tuolta", "tuolle", "tuona", "tuoksi",
    "se", "sen", "sitä", "siinä", "siitä", "siihen", "sillä", "siltä", "sille", "siksi",
    "nämä", "näiden", "näitä", "näissä", "näistä", "näihin", "näillä", "näiltä", "näille",
    "näinä", "näiksi", "nuo", "noiden", "noita", "noissa", "noista", "noihin", "noilla",
    "noilta", "noille", "noina", "noiksi", "ne", "niiden", "niitä", "niissä", "niistä",
    "niihin", "niillä", "niiltä", "niille", "niinä", "niiksi", "kuka", "kenen", "kenet",
    "ketä", "kenessä", "kenestä", "keneen", "kenellä", "keneltä", "kenelle", "kenenä",
    "keneksi", "ketkä", "keiden", "keitä", "keissä", "keistä", "keihin", "keillä", "keiltä",
    "keille", "keinä", "keiksi", "mikä", "minkä", "mitä", "missä", "mistä", "mihin",
    "millä", "miltä", "mille", "miksi", "mitkä", "joka", "jonka", "jota", "jossa", "josta",
    "johon", "jolla", "jolta", "jolle", "jona", "joksi", "jotka", "joiden", "joita",
    "joissa", "joista", "joihin", "joilla", "joilta", "joille", "joina", "joiksi", "että",
    "ja", "jos", "koska", "kuin", "mutta", "niin", "sekä", "tai", "vaan", "vai", "vaikka",
    "kanssa", "mukaan", "noin", "poikki", "yli", "kun", "nyt", "itse",
];

/// Lucene `FrenchAnalyzer` default set: `snowball/french_stop.txt` (154 words).
pub const FRENCH: &[&str] = &[
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux",
    "il", "je", "la", "le", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi", "mon",
    "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui",
    "sa", "se", "ses", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos",
    "votre", "vous", "c", "d", "j", "l", "à", "m", "n", "s", "t", "y", "étée", "étées",
    "étant", "suis", "es", "êtes", "sont", "serai", "seras", "sera", "serons", "serez",
    "seront", "serais", "serait", "serions", "seriez", "seraient", "étais", "était",
    "étions", "étiez", "étaient", "fus", "fut", "fûmes", "fûtes", "furent", "sois", "soit",
    "soyons", "soyez", "soient", "fusse", "fusses", "fussions", "fussiez", "fussent",
    "ayant", "eu", "eue", "eues", "eus", "ai", "avons", "avez", "ont", "aurai", "aurons",
    "aurez", "auront", "aurais", "aurait", "aurions", "auriez", "auraient", "avais",
    "avait", "aviez", "avaient", "eut", "eûmes", "eûtes", "eurent", "aie", "aies", "ait",
    "ayons", "ayez", "aient", "eusse", "eusses", "eût", "eussions", "eussiez", "eussent",
    "ceci", "cela", "celà", "cet", "cette", "ici", "ils", "les", "leurs", "quel", "quels",
    "quelle", "quelles", "sans", "soi",
];

/// Lucene `GermanAnalyzer` default set: `snowball/german_stop.txt` (231 words).
pub const GERMAN: &[&str] = &[
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander",
    "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr",
    "anders", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann",
    "der", "den", "des", "dem", "die", "das", "daß", "derselbe", "derselben", "denselben",
    "desselben", "demselben", "dieselbe", "dieselben", "dasselbe", "dazu", "dein", "deine",
    "deinem", "deinen", "deiner", "deines", "denn", "derer", "dessen", "dich", "dir", "du",
    "dies", "diese", "diesem", "diesen", "dieser", "dieses", "doch", "dort", "durch", "ein",
    "eine", "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen",
    "einiger", "einiges", "einmal", "er", "ihn", "ihm", "es", "etwas", "euer", "eure",
    "eurem", "euren", "eurer", "eures", "für", "gegen", "gewesen", "hab", "habe", "haben",
    "hat", "hatte", "hatten", "hier", "hin", "hinter", "ich", "mich", "mir", "ihr", "ihre",
    "ihrem", "ihren", "ihrer", "ihres", "euch", "im", "in", "indem", "ins", "ist", "jede",
    "jedem", "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener", "jenes", "jetzt",
    "kann", "kein", "keine", "keinem", "keinen", "keiner", "keines", "können", "könnte",
    "machen", "man", "manche", "manchem", "manchen", "mancher", "manches", "mein", "meine",
    "meinem", "meinen", "meiner", "meines", "mit", "muss", "musste", "nach", "nicht",
    "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem",
    "seinen", "seiner", "seines", "selbst", "sich", "sie", "ihnen", "sind", "so", "solche",
    "solchem", "solchen", "solcher", "solches", "soll", "sollte", "sondern", "sonst",
    "über", "um", "und", "uns", "unse", "unsem", "unsen", "unser", "unses", "unter", "viel",
    "vom", "von", "vor", "während", "war", "waren", "warst", "was", "weg", "weil", "weiter",
    "welche", "welchem", "welchen", "welcher", "welches", "wenn", "werde", "werden", "wie",
    "wieder", "will", "wir", "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden",
    "zu", "zum", "zur", "zwar", "zwischen",
];

/// Lucene `HungarianAnalyzer` default set: `snowball/hungarian_stop.txt` (198 words).
pub const HUNGARIAN: &[&str] = &[
    "a", "ahogy", "ahol", "aki", "akik", "akkor", "alatt", "által", "általában", "amely",
    "amelyek", "amelyekben", "amelyeket", "amelyet", "amelynek", "ami", "amit", "amolyan",
    "amíg", "amikor", "át", "abban", "ahhoz", "annak", "arra", "arról", "az", "azok",
    "azon", "azt", "azzal", "azért", "aztán", "azután", "azonban", "bár", "be", "belül",
    "benne", "cikk", "cikkek", "cikkeket", "csak", "de", "e", "eddig", "egész", "egy",
    "egyes", "egyetlen", "egyéb", "egyik", "egyre", "ekkor", "el", "elég", "ellen", "elő",
    "először", "előtt", "első", "én", "éppen", "ebben", "ehhez", "emilyen", "ennek", "erre",
    "ez", "ezt", "ezek", "ezen", "ezzel", "ezért", "és", "fel", "felé", "hanem", "hiszen",
    "hogy", "hogyan", "igen", "így", "illetve", "ill.", "ill", "ilyen", "ilyenkor", "ison",
    "ismét", "itt", "jó", "jól", "jobban", "kell", "kellett", "keresztül", "keressünk",
    "ki", "kívül", "között", "közül", "legalább", "lehet", "lehetett", "legyen", "lenne",
    "lenni", "lesz", "lett", "maga", "magát", "majd", "már", "más", "másik", "meg", "még",
    "mellett", "mert", "mely", "melyek", "mi", "mit", "míg", "miért", "milyen", "mikor",
    "minden", "mindent", "mindenki", "mindig", "mint", "mintha", "mivel", "most", "nagy",
    "nagyobb", "nagyon", "ne", "néha", "nekem", "neki", "nem", "néhány", "nélkül", "nincs",
    "olyan", "ott", "össze", "ő", "ők", "őket", "pedig", "persze", "rá", "s", "saját",
    "sem", "semmi", "sok", "sokat", "sokkal", "számára", "szemben", "szerint", "szinte",
    "talán", "tehát", "teljes", "tovább", "továbbá", "több", "úgy", "ugyanis", "új",
    "újabb", "újra", "után", "utána", "utolsó", "vagy", "vagyis", "valaki", "valami",
    "valamint", "való", "vagyok", "van", "vannak", "volt", "voltam", "voltak", "voltunk",
    "vissza", "vele", "viszont", "volna",
];

/// Lucene `ItalianAnalyzer` default set: `snowball/italian_stop.txt` (279 words).
pub const ITALIAN: &[&str] = &[
    "ad", "al", "allo", "ai", "agli", "all", "agl", "alla", "alle", "con", "col", "coi",
    "da", "dal", "dallo", "dai", "dagli", "dall", "dagl", "dalla", "dalle", "di", "del",
    "dello", "dei", "degli", "dell", "degl", "della", "delle", "in", "nel", "nello", "nei",
    "negli", "nell", "negl", "nella", "nelle", "su", "sul", "sullo", "sui", "sugli", "sull",
    "sugl", "sulla", "sulle", "per", "tra", "contro", "io", "tu", "lui", "lei", "noi",
    "voi", "loro", "mio", "mia", "miei", "mie", "tuo", "tua", "tuoi", "tue", "suo", "sua",
    "suoi", "sue", "nostro", "nostra", "nostri", "nostre", "vostro", "vostra", "vostri",
    "vostre", "mi", "ti", "ci", "vi", "lo", "la", "li", "le", "gli", "ne", "il", "un",
    "uno", "una", "ma", "ed", "se", "perché", "anche", "come", "dov", "dove", "che", "chi",
    "cui", "non", "più", "quale", "quanto", "quanti", "quanta", "quante", "quello",
    "quelli", "quella", "quelle", "questo", "questi", "questa", "queste", "si", "tutto",
    "tutti", "a", "c", "e", "i", "l", "o", "ho", "hai", "ha", "abbiamo", "avete", "hanno",
    "abbia", "abbiate", "abbiano", "avrò", "avrai", "avrà", "avremo", "avrete", "avranno",
    "avrei", "avresti", "avrebbe", "avremmo", "avreste", "avrebbero", "avevo", "avevi",
    "aveva", "avevamo", "avevate", "avevano", "ebbi", "avesti", "ebbe", "avemmo", "aveste",
    "ebbero", "avessi", "avesse", "avessimo", "avessero", "avendo", "avuto", "avuta",
    "avuti", "avute", "sono", "sei", "è", "siamo", "siete", "sia", "siate", "siano", "sarò",
    "sarai", "sarà", "saremo", "sarete", "saranno", "sarei", "saresti", "sarebbe",
    "saremmo", "sareste", "sarebbero", "ero", "eri", "era", "eravamo", "eravate", "erano",
    "fui", "fosti", "fu", "fummo", "foste", "furono", "fossi", "fosse", "fossimo",
    "fossero", "essendo", "faccio", "fai", "facciamo", "fanno", "faccia", "facciate",
    "facciano", "farò", "farai", "farà", "faremo", "farete", "faranno", "farei", "faresti",
    "farebbe", "faremmo", "fareste", "farebbero", "facevo", "facevi", "faceva", "facevamo",
    "facevate", "facevano", "feci", "facesti", "fece", "facemmo", "faceste", "fecero",
    "facessi", "facesse", "facessimo", "facessero", "facendo", "sto", "stai", "sta",
    "stiamo", "stanno", "stia", "stiate", "stiano", "starò", "starai", "starà", "staremo",
    "starete", "staranno", "starei", "staresti", "starebbe", "staremmo", "stareste",
    "starebbero", "stavo", "stavi", "stava", "stavamo", "stavate", "stavano", "stetti",
    "stesti", "stette", "stemmo", "steste", "stettero", "stessi", "stesse", "stessimo",
    "stessero", "stando",
];

/// Lucene `NorwegianAnalyzer` default set: `snowball/norwegian_stop.txt` (172 words).
pub const NORWEGIAN: &[&str] = &[
    "og", "i", "jeg", "det", "at", "en", "et", "den", "til", "er", "som", "på", "de", "med",
    "han", "av", "ikke", "ikkje", "der", "så", "var", "meg", "seg", "men", "ett", "har",
    "om", "vi", "min", "mitt", "ha", "hadde", "hun", "nå", "over", "da", "ved", "fra", "du",
    "ut", "sin", "dem", "oss", "opp", "man", "kan", "hans", "hvor", "eller", "hva", "skal",
    "selv", "sjøl", "her", "alle", "vil", "bli", "ble", "blei", "blitt", "kunne", "inn",
    "når", "være", "kom", "noen", "noe", "ville", "dere", "deres", "kun", "ja", "etter",
    "ned", "skulle", "denne", "for", "deg", "si", "sine", "sitt", "mot", "å", "meget",
    "hvorfor", "dette", "disse", "uten", "hvordan", "ingen", "din", "ditt", "blir", "samme",
    "hvilken", "hvilke", "sånn", "inni", "mellom", "vår", "hver", "hvem", "vors", "hvis",
    "både", "bare", "enn", "fordi", "før", "mange", "også", "slik", "vært", "båe", "begge",
    "siden", "dykk", "dykkar", "dei", "deira", "deires", "deim", "di", "då", "eg", "ein",
    "eit", "eitt", "elles", "honom", "hjå", "ho", "hoe", "henne", "hennar", "hennes",
    "hoss", "hossen", "ingi", "inkje", "korleis", "korso", "kva", "kvar", "kvarhelst",
    "kven", "kvi", "kvifor", "me", "medan", "mi", "mine", "mykje", "no", "nokon", "noka",
    "nokor", "noko", "nokre", "sia", "sidan", "so", "somt", "somme", "um", "upp", "vere",
    "vore", "verte", "vort", "varte", "vart",
];

/// Lucene `PortugueseAnalyzer` default set: `snowball/portuguese_stop.txt` (203 words).
pub const PORTUGUESE: &[&str] = &[
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os",
    "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele", "das", "à",
    "seu", "sua", "ou", "quando", "muito", "nos", "já", "eu", "também", "só", "pelo",
    "pela", "até", "isso", "ela", "entre", "depois", "sem", "mesmo", "aos", "seus", "quem",
    "nas", "me", "esse", "eles", "você", "essa", "num", "nem", "suas", "meu", "às", "minha",
    "numa", "pelos", "elas", "qual", "nós", "lhe", "deles", "essas", "esses", "pelas",
    "este", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua",
    "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes",
    "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "está",
    "estamos", "estão", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos",
    "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam", "estivesse",
    "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "há",
    "havemos", "hão", "houve", "houvemos", "houveram", "houvera", "houvéramos", "haja",
    "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem", "houver", "houvermos",
    "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria", "houveríamos",
    "houveriam", "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi", "fomos",
    "foram", "fora", "fôramos", "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem",
    "for", "formos", "forem", "serei", "será", "seremos", "serão", "seria", "seríamos",
    "seriam", "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham", "tive", "teve",
    "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos", "tenham", "tivesse",
    "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos",
    "terão", "teria", "teríamos", "teriam",
];

/// Lucene `RussianAnalyzer` default set: `snowball/russian_stop.txt` (159 words).
pub const RUSSIAN: &[&str] = &[
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
    "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
    "вас", "нибудь", "опять", "уж", "вам", "сказал", "ведь", "там", "потом", "себя",
    "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы",
    "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "человек", "чего", "раз",
    "тоже", "себе", "под", "жизнь", "будет", "ж", "тогда", "кто", "этот", "говорил", "того",
    "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "кажется", "сейчас", "были", "куда", "зачем", "сказать", "всех",
    "никогда", "сегодня", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
    "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "сказала", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед",
    "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно",
    "всю", "между",
];

/// Lucene `SpanishAnalyzer` default set: `snowball/spanish_stop.txt` (308 words).
pub const SPANISH: &[&str] = &[
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o",
    "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también",
    "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí",
    "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa",
    "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas",
    "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas",
    "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya",
    "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros",
    "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy",
    "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos", "estéis",
    "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría",
    "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos",
    "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis",
    "estuvieron", "estuviera", "estuvieras", "estuviéramos", "estuvierais", "estuvieran",
    "estuviese", "estuvieses", "estuviésemos", "estuvieseis", "estuviesen", "estando",
    "estado", "estada", "estados", "estadas", "estad", "he", "has", "ha", "hemos", "habéis",
    "han", "haya", "hayas", "hayamos", "hayáis", "hayan", "habré", "habrás", "habrá",
    "habremos", "habréis", "habrán", "habría", "habrías", "habríamos", "habríais",
    "habrían", "había", "habías", "habíamos", "habíais", "habían", "hube", "hubiste",
    "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubiéramos",
    "hubierais", "hubieran", "hubiese", "hubieses", "hubiésemos", "hubieseis", "hubiesen",
    "habiendo", "habido", "habida", "habidos", "habidas", "soy", "eres", "es", "somos",
    "sois", "son", "sea", "seas", "seamos", "seáis", "sean", "seré", "serás", "será",
    "seremos", "seréis", "serán", "sería", "serías", "seríamos", "seríais", "serían", "era",
    "eras", "éramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos", "fuisteis",
    "fueron", "fuera", "fueras", "fuéramos", "fuerais", "fueran", "fuese", "fueses",
    "fuésemos", "fueseis", "fuesen", "siendo", "sido", "tengo", "tienes", "tiene",
    "tenemos", "tenéis", "tienen", "tenga", "tengas", "tengamos", "tengáis", "tengan",
    "tendré", "tendrás", "tendrá", "tendremos", "tendréis", "tendrán", "tendría",
    "tendrías", "tendríamos", "tendríais", "tendrían", "tenía", "tenías", "teníamos",
    "teníais", "tenían", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron",
    "tuviera", "tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese", "tuvieses",
    "tuviésemos", "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida", "tenidos",
    "tenidas", "tened",
];

/// Lucene `SwedishAnalyzer` default set: `snowball/swedish_stop.txt` (114 words).
pub const SWEDISH: &[&str] = &[
    "och", "det", "att", "i", "en", "jag", "hon", "som", "han", "på", "den", "med", "var",
    "sig", "för", "så", "till", "är", "men", "ett", "om", "hade", "de", "av", "icke", "mig",
    "du", "henne", "då", "sin", "nu", "har", "inte", "hans", "honom", "skulle", "hennes",
    "där", "min", "man", "ej", "vid", "kunde", "något", "från", "ut", "när", "efter", "upp",
    "vi", "dem", "vara", "vad", "över", "än", "dig", "kan", "sina", "här", "ha", "mot",
    "alla", "under", "någon", "eller", "allt", "mycket", "sedan", "ju", "denna", "själv",
    "detta", "åt", "utan", "varit", "hur", "ingen", "mitt", "ni", "bli", "blev", "oss",
    "din", "dessa", "några", "deras", "blir", "mina", "samma", "vilken", "er", "sådan",
    "vår", "blivit", "dess", "inom", "mellan", "sådant", "varför", "varje", "vilka", "ditt",
    "vem", "vilket", "sitt", "sådana", "vart", "dina", "vars", "vårt", "våra", "ert", "era",
    "vilkas",
];

/// Lucene `IrishAnalyzer` default set: `snowball/irish_stop.txt` (109 words).
pub const IRISH: &[&str] = &[
    "a", "ach", "ag", "agus", "an", "aon", "ar", "arna", "as", "b'", "ba", "beirt", "bhúr",
    "caoga", "ceathair", "ceathrar", "chomh", "chtó", "chuig", "chun", "cois", "céad",
    "cúig", "cúigear", "d'", "daichead", "dar", "de", "deich", "deichniúr", "den", "dhá",
    "do", "don", "dtí", "dá", "dár", "dó", "faoi", "faoin", "faoina", "faoinár", "fara",
    "fiche", "gach", "gan", "go", "gur", "haon", "hocht", "i", "iad", "idir", "in", "ina",
    "ins", "inár", "is", "le", "leis", "lena", "lenár", "m'", "mar", "mo", "mé", "na",
    "nach", "naoi", "naonúr", "ná", "ní", "níor", "nó", "nócha", "ocht", "ochtar", "os",
    "roimh", "sa", "seacht", "seachtar", "seachtó", "seasca", "seisear", "siad", "sibh",
    "sinn", "sna", "sé", "sí", "tar", "thar", "thú", "triúr", "trí", "trína", "trínár",
    "tríocha", "tú", "um", "ár", "é", "éis", "í", "ó", "ón", "óna", "ónár",
];

/// Lucene `ArabicAnalyzer` default set: `ar/stopwords.txt` (119 words).
pub const ARABIC: &[&str] = &[
    "من", "ومن", "منها", "منه", "في", "وفي", "فيها", "فيه", "و", "ف", "ثم", "او", "أو", "ب",
    "بها", "به", "ا", "أ", "اى", "اي", "أي", "أى", "لا", "ولا", "الا", "ألا", "إلا", "لكن",
    "ما", "وما", "كما", "فما", "عن", "مع", "اذا", "إذا", "ان", "أن", "إن", "انها", "أنها",
    "إنها", "انه", "أنه", "إنه", "بان", "بأن", "فان", "فأن", "وان", "وأن", "وإن", "التى",
    "التي", "الذى", "الذي", "الذين", "الى", "الي", "إلى", "إلي", "على", "عليها", "عليه",
    "اما", "أما", "إما", "ايضا", "أيضا", "كل", "وكل", "لم", "ولم", "لن", "ولن", "هى", "هي",
    "هو", "وهى", "وهي", "وهو", "فهى", "فهي", "فهو", "انت", "أنت", "لك", "لها", "له", "هذه",
    "هذا", "تلك", "ذلك", "هناك", "كانت", "كان", "يكون", "تكون", "وكانت", "وكان", "غير",
    "بعض", "قد", "نحو", "بين", "بينما", "منذ", "ضمن", "حيث", "الان", "الآن", "خلال", "بعد",
    "قبل", "حتى", "عند", "عندما", "لدى", "جميع",
];

/// Lucene `ArmenianAnalyzer` default set: `hy/stopwords.txt` (45 words).
pub const ARMENIAN: &[&str] = &[
    "այդ", "այլ", "այն", "այս", "դու", "դուք", "եմ", "են", "ենք", "ես", "եք", "է", "էի",
    "էին", "էինք", "էիր", "էիք", "էր", "ըստ", "թ", "ի", "ին", "իսկ", "իր", "կամ", "համար",
    "հետ", "հետո", "մենք", "մեջ", "մի", "ն", "նա", "նաև", "նրա", "նրանք", "որ", "որը",
    "որոնք", "որպես", "ու", "ում", "պիտի", "վրա", "և",
];

/// Lucene `BasqueAnalyzer` default set: `eu/stopwords.txt` (98 words).
pub const BASQUE: &[&str] = &[
    "al", "anitz", "arabera", "asko", "baina", "bat", "batean", "batek", "bati", "batzuei",
    "batzuek", "batzuetan", "batzuk", "bera", "beraiek", "berau", "berauek", "bere",
    "berori", "beroriek", "beste", "bezala", "da", "dago", "dira", "ditu", "du", "dute",
    "edo", "egin", "ere", "eta", "eurak", "ez", "gainera", "gu", "gutxi", "guzti", "haiei",
    "haiek", "haietan", "hainbeste", "hala", "han", "handik", "hango", "hara", "hari",
    "hark", "hartan", "hau", "hauei", "hauek", "hauetan", "hemen", "hemendik", "hemengo",
    "hi", "hona", "honek", "honela", "honetan", "honi", "hor", "hori", "horiei", "horiek",
    "horietan", "horko", "horra", "horrek", "horrela", "horretan", "horri", "hortik",
    "hura", "izan", "ni", "noiz", "nola", "non", "nondik", "nongo", "nor", "nora", "ze",
    "zein", "zen", "zenbait", "zenbat", "zer", "zergatik", "ziren", "zituen", "zu", "zuek",
    "zuen", "zuten",
];

/// Lucene `BengaliAnalyzer` default set: `bn/stopwords.txt` (116 words).
pub const BENGALI: &[&str] = &[
    "এই", "ও", "থেকে", "করে", "এ", "না", "ওই", "এক্", "নিয়ে", "করা", "বলেন", "সঙ্গে", "যে",
    "এব", "তা", "আর", "কোনো", "বলে", "সেই", "দিন", "হয়", "কি", "দু", "পরে", "সব", "দেওয়া",
    "মধ্যে", "এর", "সি", "শুরু", "কাজ", "কিছু", "কাছে", "সে", "তবে", "বা", "বন", "আগে",
    "জ্নজন", "পি", "পর", "তো", "ছিল", "এখন", "আমরা", "প্রায়", "দুই", "আমাদের", "তাই",
    "অন্য", "গিয়ে", "প্রযন্ত", "মনে", "নতুন", "মতো", "কেখা", "প্রথম", "আজ", "টি", "ধামার",
    "অনেক", "বিভিন্ন", "র", "হাজার", "জানা", "নয়", "অবশ্য", "বেশি", "এস", "কে", "হতে", "বি",
    "কয়েক", "সহ", "বেশ", "এমন", "এমনি", "কেন", "কেউ", "নেওয়া", "চেষ্টা", "লক্ষ", "বলা",
    "কারণ", "আছে", "শুধু", "তখন", "যা", "এসে", "চার", "যদি", "আবার", "কোটি", "উত্তর",
    "সামনে", "উপর", "বক্তব্য", "এত", "প্রাথমিক", "উপরে", "প্রতি", "কাজে", "যখন", "খুব",
    "বহু", "গেল", "পেয়্র্", "চালু", "ই", "নাগাদ", "থাকা", "পাচ", "যাওয়া", "রকম", "সাধারণ",
    "কমনে",
];

/// Lucene `BrazilianAnalyzer` default set: `br/stopwords.txt` (127 words).
///
/// Lucene `main` (apache/lucene#14075, Dec 2024) replaced a duplicated
/// `nas` with `no`; released Lucene 10.x (and so Elasticsearch) still lacks
/// `no` (126 words). All other lists are identical between `main` and
/// `branch_10x`.
pub const BRAZILIAN: &[&str] = &[
    "a", "ainda", "alem", "ambas", "ambos", "antes", "ao", "aonde", "aos", "apos", "aquele",
    "aqueles", "as", "assim", "com", "como", "contra", "contudo", "cuja", "cujas", "cujo",
    "cujos", "da", "das", "de", "dela", "dele", "deles", "demais", "depois", "desde",
    "desta", "deste", "dispoe", "dispoem", "diversa", "diversas", "diversos", "do", "dos",
    "durante", "e", "ela", "elas", "ele", "eles", "em", "entao", "entre", "essa", "essas",
    "esse", "esses", "esta", "estas", "este", "estes", "ha", "isso", "isto", "logo", "mais",
    "mas", "mediante", "menos", "mesma", "mesmas", "mesmo", "mesmos", "na", "no", "nao",
    "nas", "nem", "nesse", "neste", "nos", "o", "os", "ou", "outra", "outras", "outro",
    "outros", "pelas", "pelo", "pelos", "perante", "pois", "por", "porque", "portanto",
    "proprio", "propios", "quais", "qual", "qualquer", "quando", "quanto", "que", "quem",
    "quer", "se", "seja", "sem", "sendo", "seu", "seus", "sob", "sobre", "sua", "suas",
    "tal", "tambem", "teu", "teus", "toda", "todas", "todo", "todos", "tua", "tuas", "tudo",
    "um", "uma", "umas", "uns",
];

/// Lucene `BulgarianAnalyzer` default set: `bg/stopwords.txt` (190 words).
pub const BULGARIAN: &[&str] = &[
    "а", "аз", "ако", "ала", "бе", "без", "беше", "би", "бил", "била", "били", "било",
    "близо", "бъдат", "бъде", "бяха", "в", "вас", "ваш", "ваша", "вероятно", "вече",
    "взема", "ви", "вие", "винаги", "все", "всеки", "всички", "всичко", "всяка", "във",
    "въпреки", "върху", "г", "ги", "главно", "го", "д", "да", "дали", "до", "докато",
    "докога", "дори", "досега", "доста", "е", "едва", "един", "ето", "за", "зад", "заедно",
    "заради", "засега", "затова", "защо", "защото", "и", "из", "или", "им", "има", "имат",
    "иска", "й", "каза", "как", "каква", "какво", "както", "какъв", "като", "кога",
    "когато", "което", "които", "кой", "който", "колко", "която", "къде", "където", "към",
    "ли", "м", "ме", "между", "мен", "ми", "мнозина", "мога", "могат", "може", "моля",
    "момента", "му", "н", "на", "над", "назад", "най", "направи", "напред", "например",
    "нас", "не", "него", "нея", "ни", "ние", "никой", "нито", "но", "някои", "някой",
    "няма", "обаче", "около", "освен", "особено", "от", "отгоре", "отново", "още", "пак",
    "по", "повече", "повечето", "под", "поне", "поради", "после", "почти", "прави", "пред",
    "преди", "през", "при", "пък", "първо", "с", "са", "само", "се", "сега", "си", "скоро",
    "след", "сме", "според", "сред", "срещу", "сте", "съм", "със", "също", "т", "тази",
    "така", "такива", "такъв", "там", "твой", "те", "тези", "ти", "тн", "то", "това",
    "тогава", "този", "той", "толкова", "точно", "трябва", "тук", "тъй", "тя", "тях", "у",
    "харесва", "ч", "че", "често", "чрез", "ще", "щом", "я",
];

/// Lucene `CatalanAnalyzer` default set: `ca/stopwords.txt` (218 words).
pub const CATALAN: &[&str] = &[
    "a", "abans", "ací", "ah", "així", "això", "al", "als", "aleshores", "algun", "alguna",
    "algunes", "alguns", "alhora", "allà", "allí", "allò", "altra", "altre", "altres",
    "amb", "ambdós", "ambdues", "apa", "aquell", "aquella", "aquelles", "aquells", "aquest",
    "aquesta", "aquestes", "aquests", "aquí", "baix", "cada", "cadascú", "cadascuna",
    "cadascunes", "cadascuns", "com", "contra", "d'un", "d'una", "d'unes", "d'uns", "dalt",
    "de", "del", "dels", "des", "després", "dins", "dintre", "donat", "doncs", "durant",
    "e", "eh", "el", "els", "em", "en", "encara", "ens", "entre", "érem", "eren", "éreu",
    "es", "és", "esta", "està", "estàvem", "estaven", "estàveu", "esteu", "et", "etc",
    "ets", "fins", "fora", "gairebé", "ha", "han", "has", "havia", "he", "hem", "heu", "hi",
    "ho", "i", "igual", "iguals", "ja", "l'hi", "la", "les", "li", "li'n", "llavors",
    "m'he", "ma", "mal", "malgrat", "mateix", "mateixa", "mateixes", "mateixos", "me",
    "mentre", "més", "meu", "meus", "meva", "meves", "molt", "molta", "moltes", "molts",
    "mon", "mons", "n'he", "n'hi", "ne", "ni", "no", "nogensmenys", "només", "nosaltres",
    "nostra", "nostre", "nostres", "o", "oh", "oi", "on", "pas", "pel", "pels", "per",
    "però", "perquè", "poc", "poca", "pocs", "poques", "potser", "propi", "qual", "quals",
    "quan", "quant", "que", "què", "quelcom", "qui", "quin", "quina", "quines", "quins",
    "s'ha", "s'han", "sa", "semblant", "semblants", "ses", "seu", "seus", "seva", "seves",
    "si", "sobre", "sobretot", "sóc", "solament", "sols", "son", "són", "sons", "sota",
    "sou", "t'ha", "t'han", "t'he", "ta", "tal", "també", "tampoc", "tan", "tant", "tanta",
    "tantes", "teu", "teus", "teva", "teves", "ton", "tons", "tot", "tota", "totes", "tots",
    "un", "una", "unes", "uns", "us", "va", "vaig", "vam", "van", "vas", "veu", "vosaltres",
    "vostra", "vostre", "vostres",
];

/// Lucene `CJKAnalyzer` default set: `cjk/stopwords.txt` (35 words).
pub const CJK: &[&str] = &[
    "a", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "s", "such", "t", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with", "www",
];

/// Lucene `CzechAnalyzer` default set: `cz/stopwords.txt` (171 words).
pub const CZECH: &[&str] = &[
    "a", "s", "k", "o", "i", "u", "v", "z", "dnes", "cz", "tímto", "budeš", "budem", "byli",
    "jseš", "můj", "svým", "ta", "tomto", "tohle", "tuto", "tyto", "jej", "zda", "proč",
    "máte", "tato", "kam", "tohoto", "kdo", "kteří", "mi", "nám", "tom", "tomuto", "mít",
    "nic", "proto", "kterou", "byla", "toho", "protože", "asi", "ho", "naši", "napište",
    "re", "což", "tím", "takže", "svých", "její", "svými", "jste", "aj", "tu", "tedy",
    "teto", "bylo", "kde", "ke", "pravé", "ji", "nad", "nejsou", "či", "pod", "téma",
    "mezi", "přes", "ty", "pak", "vám", "ani", "když", "však", "neg", "jsem", "tento",
    "článku", "články", "aby", "jsme", "před", "pta", "jejich", "byl", "ještě", "až", "bez",
    "také", "pouze", "první", "vaše", "která", "nás", "nový", "tipy", "pokud", "může",
    "strana", "jeho", "své", "jiné", "zprávy", "nové", "není", "vás", "jen", "podle", "zde",
    "už", "být", "více", "bude", "již", "než", "který", "by", "které", "co", "nebo", "ten",
    "tak", "má", "při", "od", "po", "jsou", "jak", "další", "ale", "si", "se", "ve", "to",
    "jako", "za", "zpět", "ze", "do", "pro", "je", "na", "atd", "atp", "jakmile", "přičemž",
    "já", "on", "ona", "ono", "oni", "ony", "my", "vy", "jí", "mě", "mne", "jemu", "tomu",
    "těm", "těmu", "němu", "němuž", "jehož", "jíž", "jelikož", "jež", "jakož", "načež",
];

/// Lucene `EstonianAnalyzer` default set: `et/stopwords.txt` (1470 words).
pub const ESTONIAN: &[&str] = &[
    "all", "alla", "allapoole", "allpool", "alt", "altpoolt", "eel", "eespool", "enne",
    "hommikupoole", "hoolimata", "ilma", "kaudu", "keset", "kesk", "kohe", "koos",
    "kuhupoole", "kuni", "kuspool", "kustpoolt", "kõige", "käsikäes", "lappi", "ligi",
    "läbi", "mööda", "paitsi", "peale", "pealepoole", "pealpool", "pealt", "pealtpoolt",
    "piki", "pikku", "piku", "pikuti", "põiki", "pärast", "päri", "risti", "sealpool",
    "sealtpoolt", "seespool", "seltsis", "siiapoole", "siinpool", "siitpoolt", "sinnapoole",
    "sissepoole", "taga", "tagantpoolt", "tagapidi", "tagapool", "taha", "tahapoole",
    "teispool", "teispoole", "tänu", "tükkis", "vaatamata", "vastu", "väljapoole",
    "väljaspool", "väljastpoolt", "õhtupoole", "ühes", "ühestükis", "ühestükkis",
    "ülalpool", "ülaltpoolt", "üle", "ülespoole", "ülevalpool", "ülevaltpoolt", "ümber",
    "ümbert", "aegu", "aegus", "alguks", "algul", "algule", "algult", "alguni", "alul",
    "alutsi", "arvel", "asemel", "asemele", "eeli", "ees", "eesotsas", "eest", "eestotsast",
    "esitsi", "ette", "etteotsa", "haaval", "heaks", "hulgas", "hulgast", "hulka", "jalgu",
    "jalus", "jalust", "jaoks", "jooksul", "juurde", "juures", "juurest", "jälil", "jälile",
    "järel", "järele", "järelt", "järgi", "kaasas", "kallal", "kallale", "kallalt", "kamul",
    "kannul", "kannule", "kannult", "kaupa", "keskel", "keskele", "keskelt", "keskis",
    "keskpaiku", "kestel", "kestes", "kilda", "killas", "killast", "kimpu", "kimpus",
    "kiuste", "kohal", "kohale", "kohalt", "kohaselt", "kohta", "korral", "kukil", "kukile",
    "kukilt", "kulul", "kõrva", "kõrval", "kõrvale", "kõrvalt", "kõrvas", "kõrvast",
    "käekõrval", "käekõrvale", "käekõrvalt", "käes", "käest", "kätte", "külge", "küljes",
    "küljest", "küüsi", "küüsis", "küüsist", "ligidal", "ligidale", "ligidalt", "lool",
    "lähedal", "lähedale", "lähedalt", "man", "mant", "manu", "meelest", "nahas", "nahka",
    "nahkas", "najal", "najale", "najalt", "nõjal", "nõjale", "otsa", "otsas", "otsast",
    "paigale", "paigu", "paiku", "peal", "perra", "perrä", "pidi", "pihta", "pool", "poole",
    "poolest", "poolt", "puhul", "puksiiris", "pähe", "päralt", "päras", "ringi", "ringis",
    "risust", "saadetusel", "saadik", "saatel", "saati", "seas", "seast", "sees", "seest",
    "sekka", "seljataga", "seltsi", "seltsist", "sisse", "slepis", "suhtes", "šlepis",
    "tagant", "tagantotsast", "tagaotsas", "tagaselja", "tagasi", "tagast", "tagutsi",
    "tahaotsa", "takka", "tarvis", "tasa", "tuuri", "tuuris", "tõttu", "uhal", "vahel",
    "vahele", "vahelt", "vahepeal", "vahepeale", "vahepealt", "vahetsi", "varal", "varale",
    "varul", "vastas", "vastast", "veerde", "veeres", "viisi", "võidu", "võrd", "võrdki",
    "võrra", "võrragi", "väel", "väele", "vältel", "väärt", "väärtki", "äärde", "ääre",
    "ääres", "äärest", "a", "abil", "aina", "ainult", "alalt", "alates", "alati", "alles",
    "b", "c", "d", "e", "eales", "ealeski", "edasi", "edaspidi", "eelkõige", "eemal", "ei",
    "eks", "end", "enda", "enese", "ennem", "esialgu", "f", "g", "h", "hoopis", "i",
    "iganes", "igatahes", "igati", "iial", "iialgi", "ikka", "ikkagi", "ilmaski", "iseenda",
    "iseenese", "iseenesest", "isegi", "j", "jah", "ju", "juba", "juhul", "just",
    "järelikult", "k", "ka", "kah", "kas", "kasvõi", "keda", "kestahes", "kogu", "koguni",
    "kohati", "kokku", "kuhu", "kuhugi", "kuidagi", "kuidas", "kunagi", "kus", "kusagil",
    "kusjuures", "kuskil", "kust", "kõigepealt", "küll", "l", "liiga", "lisaks", "m",
    "miks", "mil", "millal", "millalgi", "mispärast", "mistahes", "mistõttu", "mitte",
    "muide", "muidu", "muidugi", "muist", "mujal", "mujale", "mujalt", "mõlemad", "mõnda",
    "mõne", "mõnikord", "n", "nii", "niikaua", "niimoodi", "niipaljuke", "niisama",
    "niisiis", "niivõrd", "nõnda", "nüüd", "o", "omaette", "omakorda", "omavahel", "ometi",
    "p", "palju", "paljuke", "palju-palju", "peaaegu", "peagi", "peamiselt", "pigem",
    "pisut", "praegu", "päris", "r", "rohkem", "s", "samas", "samuti", "seal", "sealt",
    "sedakorda", "sedapuhku", "seega", "seejuures", "seejärel", "seekord", "seepärast",
    "seetõttu", "sellepärast", "seni", "sestap", "siia", "siiani", "siin", "siinkohal",
    "siis", "siiski", "siit", "sinna", "suht", "š", "z", "ž", "t", "teel", "teineteise",
    "tõesti", "täiesti", "u", "umbes", "v", "w", "veel", "veelgi", "vist", "võibolla",
    "võib-olla", "väga", "vähemalt", "välja", "väljas", "väljast", "õ", "ä", "ära", "ö",
    "ü", "ühtlasi", "üksi", "ükskõik", "ülal", "ülale", "ülalt", "üles", "ülesse", "üleval",
    "ülevalt", "ülimalt", "üsna", "x", "y", "aga", "ega", "ehk", "ehkki", "elik", "ellik",
    "enge", "ennegu", "ent", "et", "ja", "justkui", "kui", "kuid", "kuigi", "kuivõrd",
    "kuna", "kut", "mistab", "muudkui", "nagu", "nigu", "ning", "olgugi", "otsekui",
    "otsenagu", "selmet", "sest", "sestab", "vaid", "või", "aa", "adaa", "adjöö", "ae",
    "ah", "ahaa", "ahah", "ah-ah-ah", "ah-haa", "ahoi", "ai", "aidaa", "aidu-raidu", "aih",
    "aijeh", "aituma", "aitäh", "aitüma", "ammuu", "amps", "ampsti", "aptsih", "ass", "at",
    "ata", "at-at-at", "atsih", "atsihh", "auh", "bai-bai", "bingo", "braavo", "brr", "ee",
    "eeh", "eh", "ehee", "eheh", "eh-eh-hee", "eh-eh-ee", "ehei", "ehh", "ehhee", "einoh",
    "ena", "ennäe", "ennäh", "fuh", "fui", "fuih", "haa", "hah", "hahaa", "hah-hah-hah",
    "halleluuja", "hallo", "halloo", "hass", "hee", "heh", "he-he-hee", "hei",
    "heldeke(ne)", "heureka", "hihii", "hip-hip-hurraa", "hmh", "hmjah", "hoh-hoh-hoo",
    "hohoo", "hoi", "hollallaa", "hoo", "hoplaa", "hopp", "hops", "hopsassaa", "hopsti",
    "hosianna", "huh", "huidii", "huist", "hurjah", "hurjeh", "hurjoh", "hurjuh", "hurraa",
    "huu", "hõhõh", "hõi", "hõissa", "hõissassa", "hõk", "hõkk", "häh", "hä-hä-hää",
    "hüvasti", "ih-ah-haa", "ih-ih-hii", "ii-ha-ha", "issake", "issakene", "isver",
    "jaa-ah", "ja-ah", "jaah", "janäe", "jeeh", "jeerum", "jeever", "jessas", "jestas",
    "juhhei", "jumalaga", "jumalime", "jumaluke", "jumalukene", "jutas", "kaaps", "kaapsti",
    "kaasike", "kae", "kalps", "kalpsti", "kannäe", "kanäe", "kappadi", "kaps", "kapsti",
    "karkõmm", "karkäuh", "karkääks", "karkääksti", "karmauh", "karmauhti", "karnaps",
    "karnapsti", "karniuhti", "karpartsaki", "karpauh", "karpauhti", "karplauh",
    "karplauhti", "karprauh", "karprauhti", "karsumdi", "karsumm", "kartsumdi", "kartsumm",
    "karviuh", "karviuhti", "kaske", "kassa", "kauh", "kauhti", "keh", "keksti", "kepsti",
    "khe", "khm", "kih", "kiiks", "kiiksti", "kiis", "kiiss", "kikerii", "kikerikii",
    "kili", "kilk", "kilk-kõlk", "kilks", "kilks-kolks", "kilks-kõlks", "kill", "killadi",
    "killadi|-kolladi", "killadi-kõlladi", "killa-kolla", "killa-kõlla", "kill-kõll",
    "kimps-komps", "kipp", "kips-kõps", "kiriküüt", "kirra-kõrra", "kirr-kõrr", "kirts",
    "klaps", "klapsti", "klirdi", "klirr", "klonks", "klops", "klopsti", "kluk", "klu-kluu",
    "klõks", "klõksti", "klõmdi", "klõmm", "klõmpsti", "klõnks", "klõnksti", "klõps",
    "klõpsti", "kläu", "kohva-kohva", "kok", "koks", "koksti", "kolaki", "kolk", "kolks",
    "kolksti", "koll", "kolladi", "komp", "komps", "kompsti", "kop", "kopp", "koppadi",
    "kops", "kopsti", "kossu", "kotsu", "kraa", "kraak", "kraaks", "kraaps", "kraapsti",
    "krahh", "kraks", "kraksti", "kraps", "krapsti", "krauh", "krauhti", "kriiks",
    "kriiksti", "kriips", "kriips-kraaps", "kripa-krõpa", "krips-kraps", "kriuh", "kriuks",
    "kriuksti", "kromps", "kronk", "kronks", "krooks", "kruu", "krõks", "krõksti", "krõpa",
    "krõps", "krõpsti", "krõuh", "kräu", "kräuh", "kräuhti", "kräuks", "kss", "kukeleegu",
    "kukku", "kuku", "kulu", "kurluu", "kurnäu", "kuss", "kussu", "kõks", "kõksti", "kõldi",
    "kõlks", "kõlksti", "kõll", "kõmaki", "kõmdi", "kõmm", "kõmps", "kõpp", "kõps",
    "kõpsadi", "kõpsat", "kõpsti", "kõrr", "kõrra-kõrra", "kõss", "kõtt", "kõõksti", "kärr",
    "kärts", "kärtsti", "käuks", "käuksti", "kääga", "kääks", "kääksti", "köh", "köki-möki",
    "köksti", "laks", "laksti", "lampsti", "larts", "lartsti", "lats", "latsti", "leelo",
    "legoo", "lehva", "liiri-lõõri", "lika-lõka", "likat-lõkat", "limpsti", "lips",
    "lipsti", "lirts", "lirtsaki", "lirtsti", "lonksti", "lops", "lopsti", "lorts",
    "lortsti", "luks", "lups", "lupsti", "lurts", "lurtsti", "lõks", "lõksti", "lõmps",
    "lõmpsti", "lõnks", "lõnksti", "lärts", "lärtsti", "läts", "lätsti", "lörts", "lörtsti",
    "lötsti", "lööps", "lööpsti", "marss", "mats", "matsti", "mauh", "mauhti", "mh", "mhh",
    "mhmh", "miau", "mjaa", "mkm", "m-mh", "mnjaa", "mnjah", "moens", "mulks", "mulksti",
    "mull-mull", "mull-mull-mull", "muu", "muuh", "mõh", "mõmm", "mäh", "mäts", "mäu",
    "mää", "möh", "möh-öh-ää", "möö", "müh-müh", "mühüh", "müks", "müksti", "müraki",
    "mürr", "mürts", "mürtsaki", "mürtsti", "mütaku", "müta-mäta", "müta-müta", "müt-müt",
    "müt-müt-müt", "müts", "mütsti", "mütt", "naa", "naah", "nah", "naks", "naksti",
    "nanuu", "naps", "napsti", "nilpsti", "nipsti", "nirr", "niuh", "niuh-näuh", "niuhti",
    "noh", "noksti", "nolpsti", "nonoh", "nonoo", "nonäh", "noo", "nooh", "nooks", "norr",
    "nurr", "nuuts", "nõh", "nõhh", "nõka-nõka", "nõks", "nõksat-nõksat", "nõks-nõks",
    "nõksti", "nõõ", "nõõh", "näeh", "näh", "nälpsti", "nämm-nämm", "näpsti", "näts",
    "nätsti", "näu", "näuh", "näuhti", "näuks", "näuksti", "nääh", "nääks", "nühkat-nühkat",
    "oeh", "oh", "ohh", "ohhh", "oh-hoi", "oh-hoo", "ohoh", "oh-oh-oo", "oh-oh-hoo", "ohoi",
    "ohoo", "oi", "oih", "oijee", "oijeh", "oo", "ooh", "oo-oh", "oo-ohh", "oot", "ossa",
    "ot", "paa", "pah", "pahh", "pakaa", "pamm", "pantsti", "pardon", "pardonks",
    "parlartsti", "parts", "partsti", "partsumdi", "partsumm", "pastoi", "pats", "patst",
    "patsti", "pau", "pauh", "pauhti", "pele", "pfui", "phuh", "phuuh", "phäh", "phähh",
    "piiks", "piip", "piiri-pääri", "pimm", "pimm-pamm", "pimm-pomm", "pimm-põmm", "piraki",
    "piuks", "piu-pau", "plaks", "plaksti", "plarts", "plartsti", "plats", "platsti",
    "plauh", "plauhh", "plauhti", "pliks", "pliks-plaks", "plinn", "pliraki", "plirts",
    "plirtsti", "pliu", "pliuh", "ploks", "plotsti", "plumps", "plumpsti", "plõks",
    "plõksti", "plõmdi", "plõmm", "plõnn", "plärr", "plärts", "plärtsat", "plärtsti",
    "pläu", "pläuh", "plää", "plörtsat", "pomm", "popp", "pops", "popsti", "ports", "pot",
    "pots", "potsti", "pott", "praks", "praksti", "prants", "prantsaki", "prantsti",
    "prassai", "prauh", "prauhh", "prauhti", "priks", "priuh", "priuhh", "priuh-prauh",
    "proosit", "proost", "prr", "prrr", "prõks", "prõksti", "prõmdi", "prõmm", "prõntsti",
    "prääk", "prääks", "pst", "psst", "ptrr", "ptruu", "ptüi", "puh", "puhh", "puksti",
    "pumm", "pumps", "pup-pup-pup", "purts", "puuh", "põks", "põksti", "põmdi", "põmm",
    "põmmadi", "põnks", "põnn", "põnnadi", "põnt", "põnts", "põntsti", "põraki", "põrr",
    "põrra-põrra", "päh", "pähh", "päntsti", "pää", "pöörd", "püh", "raks", "raksti",
    "raps", "rapsti", "ratataa", "rauh", "riips", "riipsti", "riks", "riks-raks",
    "rips-raps", "rivitult", "robaki", "rops", "ropsaki", "ropsti", "ruik", "räntsti",
    "räts", "röh", "röhh", "sah", "sahh", "sahkat", "saps", "sapsti", "sauh", "sauhti",
    "servus", "sihkadi-sahkadi", "sihka-sahka", "sihkat-sahkat", "silks", "silk-solk",
    "sips", "sipsti", "sirr", "sirr-sorr", "sirts", "sirtsti", "siu", "siuh", "siuh-sauh",
    "siuh-säuh", "siuhti", "siuks", "siuts", "skool", "so", "soh", "solks", "solksti",
    "solpsti", "soo", "sooh", "so-oh", "soo-oh", "sopp", "sops", "sopsti", "sorr", "sorts",
    "sortsti", "so-soo", "soss", "soss-soss", "ss", "sss", "sst", "stopp", "suhkat-sahkat",
    "sulk", "sulks", "sulksti", "sull", "sulla-sulla", "sulpa-sulpa", "sulps", "sulpsti",
    "sumaki", "sumdi", "summ", "summat-summat", "sups", "supsaku", "supsti", "surts",
    "surtsti", "suss", "susti", "suts", "sutsti", "säh", "sähke", "särts", "särtsti", "säu",
    "säuh", "säuhti", "taevake", "taevakene", "takk", "tere", "terekest", "tibi-tibi",
    "tikk-takk", "tiks", "tilk", "tilks", "till", "tilla-talla", "till-tall", "tilulii",
    "tinn", "tip", "tip-tap", "tirr", "tirtsti", "tiu", "tjaa", "tjah", "tohhoh", "tohhoo",
    "tohoh", "tohoo", "tok", "tokk", "toks", "toksti", "tonks", "tonksti", "tota", "totsti",
    "tot-tot", "tprr", "tpruu", "trah", "trahh", "trallallaa", "trill", "trillallaa", "trr",
    "trrr", "tsah", "tsahh", "tsilk", "tsilk-tsolk", "tsirr", "tsiuh", "tskae", "tsolk",
    "tss", "tst", "tsst", "tsuhh", "tsuk", "tsumm", "tsurr", "tsäuh", "tšao", "tšš", "tššš",
    "tuk", "tuks", "turts", "turtsti", "tutki", "tutkit", "tutu-lutu", "tutulutu", "tuut",
    "tuutu-luutu", "tõks", "tötsti", "tümps", "uh", "uhh", "uh-huu", "uhtsa", "uhtsaa",
    "uhuh", "uhuu", "ui", "uih", "uih-aih", "uijah", "uijeh", "uist", "uit", "uka", "upsti",
    "uraa", "urjah", "urjeh", "urjoh", "urjuh", "urr", "urraa", "ust", "utu", "uu", "uuh",
    "vaak", "vaat", "vae", "vaeh", "vai", "vat", "vau", "vhüüt", "vidiit", "viiks", "vilks",
    "vilksti", "vinki-vinki", "virdi", "virr", "viu", "viudi", "viuh", "viuhti", "voeh",
    "voh", "vohh", "volks", "volksti", "vooh", "vops", "vopsti", "vot", "vuh", "vuhti",
    "vuih", "vulks", "vulksti", "vull", "vulpsti", "vups", "vupsaki", "vupsaku", "vupsti",
    "vurdi", "vurr", "vurra-vurra", "vurts", "vurtsti", "vutt", "võe", "võeh", "võih",
    "võrr", "võts", "võtt", "vääks", "õe", "õits", "õk", "õkk", "õrr", "õss", "õuh", "äh",
    "ähh", "ähhähhää", "äh-hää", "äh-äh-hää", "äiu", "äiu-ää", "äss", "ää", "ääh", "äähh",
    "öh", "öhh", "ök", "üh", "eelmine", "eikeegi", "eimiski", "emb-kumb", "enam", "enim",
    "iga", "igasugune", "igaüks", "ise", "isesugune", "järgmine", "keegi", "kes", "kumb",
    "kumbki", "kõik", "meiesugune", "meietaoline", "midagi", "mihuke", "mihukene",
    "milletaoline", "milline", "mina", "minake", "mingi", "mingisugune", "minusugune",
    "minutaoline", "mis", "miski", "miskisugune", "missugune", "misuke", "mitmes",
    "mitmesugune", "mitu", "mitu-mitu", "mitu-setu", "mõlema", "mõnesugune", "mõni",
    "mõningane", "mõningas", "mäherdune", "määrane", "naasugune", "need", "nemad",
    "nendesugune", "nendetaoline", "nihuke", "nihukene", "niimitu", "niisamasugune",
    "niisugune", "nisuke", "nisukene", "oma", "omaenese", "omasugune", "omataoline",
    "praegune", "sama", "samasugune", "samataoline", "see", "seesama", "seesamane",
    "seesamune", "seesinane", "seesugune", "selline", "sihuke", "sihukene", "sina",
    "sinusugune", "sinutaoline", "siuke", "siukene", "säherdune", "säärane", "taoline",
    "teiesugune", "teine", "teistsugune", "tema", "temake", "temakene", "temasugune",
    "temataoline", "too", "toosama", "toosamane", "üks", "üksteise", "hakkama", "minema",
    "olema", "pidama", "saama", "tegema", "tulema", "võima",
];

/// Lucene `GalicianAnalyzer` default set: `gl/stopwords.txt` (160 words).
pub const GALICIAN: &[&str] = &[
    "a", "aínda", "alí", "aquel", "aquela", "aquelas", "aqueles", "aquilo", "aquí", "ao",
    "aos", "as", "así", "á", "ben", "cando", "che", "co", "coa", "comigo", "con",
    "connosco", "contigo", "convosco", "coas", "cos", "cun", "cuns", "cunha", "cunhas",
    "da", "dalgunha", "dalgunhas", "dalgún", "dalgúns", "das", "de", "del", "dela", "delas",
    "deles", "desde", "deste", "do", "dos", "dun", "duns", "dunha", "dunhas", "e", "el",
    "ela", "elas", "eles", "en", "era", "eran", "esa", "esas", "ese", "eses", "esta",
    "estar", "estaba", "está", "están", "este", "estes", "estiven", "estou", "eu", "é",
    "facer", "foi", "foron", "fun", "había", "hai", "iso", "isto", "la", "las", "lle",
    "lles", "lo", "los", "mais", "me", "meu", "meus", "min", "miña", "miñas", "moi", "na",
    "nas", "neste", "nin", "no", "non", "nos", "nosa", "nosas", "noso", "nosos", "nós",
    "nun", "nunha", "nuns", "nunhas", "o", "os", "ou", "ó", "ós", "para", "pero", "pode",
    "pois", "pola", "polas", "polo", "polos", "por", "que", "se", "senón", "ser", "seu",
    "seus", "sexa", "sido", "sobre", "súa", "súas", "tamén", "tan", "te", "ten", "teñen",
    "teño", "ter", "teu", "teus", "ti", "tido", "tiña", "tiven", "túa", "túas", "un",
    "unha", "unhas", "uns", "vos", "vosa", "vosas", "voso", "vosos", "vós",
];

/// Lucene `GreekAnalyzer` default set: `el/stopwords.txt` (75 words).
pub const GREEK: &[&str] = &[
    "ο", "η", "το", "οι", "τα", "του", "τησ", "των", "τον", "την", "και", "κι", "κ",
    "ειμαι", "εισαι", "ειναι", "ειμαστε", "ειστε", "στο", "στον", "στη", "στην", "μα",
    "αλλα", "απο", "για", "προσ", "με", "σε", "ωσ", "παρα", "αντι", "κατα", "μετα", "θα",
    "να", "δε", "δεν", "μη", "μην", "επι", "ενω", "εαν", "αν", "τοτε", "που", "πωσ",
    "ποιοσ", "ποια", "ποιο", "ποιοι", "ποιεσ", "ποιων", "ποιουσ", "αυτοσ", "αυτη", "αυτο",
    "αυτοι", "αυτων", "αυτουσ", "αυτεσ", "αυτα", "εκεινοσ", "εκεινη", "εκεινο", "εκεινοι",
    "εκεινεσ", "εκεινα", "εκεινων", "εκεινουσ", "οπωσ", "ομωσ", "ισωσ", "οσο", "οτι",
];

/// Lucene `HindiAnalyzer` default set: `hi/stopwords.txt` (225 words).
pub const HINDI: &[&str] = &[
    "अंदर", "अत", "अपना", "अपनी", "अपने", "अभी", "आदि", "आप", "इत्यादि", "इन", "इनका",
    "इन्हीं", "इन्हें", "इन्हों", "इस", "इसका", "इसकी", "इसके", "इसमें", "इसी", "इसे", "उन",
    "उनका", "उनकी", "उनके", "उनको", "उन्हीं", "उन्हें", "उन्हों", "उस", "उसके", "उसी",
    "उसे", "एक", "एवं", "एस", "ऐसे", "और", "कई", "कर", "करता", "करते", "करना", "करने",
    "करें", "कहते", "कहा", "का", "काफ़ी", "कि", "कितना", "किन्हें", "किन्हों", "किया", "किर",
    "किस", "किसी", "किसे", "की", "कुछ", "कुल", "के", "को", "कोई", "कौन", "कौनसा", "गया",
    "घर", "जब", "जहाँ", "जा", "जितना", "जिन", "जिन्हें", "जिन्हों", "जिस", "जिसे", "जीधर",
    "जैसा", "जैसे", "जो", "तक", "तब", "तरह", "तिन", "तिन्हें", "तिन्हों", "तिस", "तिसे",
    "तो", "था", "थी", "थे", "दबारा", "दिया", "दुसरा", "दूसरे", "दो", "द्वारा", "न", "नहीं",
    "ना", "निहायत", "नीचे", "ने", "पर", "पहले", "पूरा", "पे", "फिर", "बनी", "बही", "बहुत",
    "बाद", "बाला", "बिलकुल", "भी", "भीतर", "मगर", "मानो", "मे", "में", "यदि", "यह", "यहाँ",
    "यही", "या", "यिह", "ये", "रखें", "रहा", "रहे", "ऱ्वासा", "लिए", "लिये", "लेकिन", "व",
    "वर्ग", "वह", "वहाँ", "वहीं", "वाले", "वुह", "वे", "वग़ैरह", "संग", "सकता", "सकते",
    "सबसे", "सभी", "साथ", "साबुत", "साभ", "सारा", "से", "सो", "ही", "हुआ", "हुई", "हुए",
    "है", "हैं", "हो", "होता", "होती", "होते", "होना", "होने", "अपनि", "जेसे", "होति",
    "सभि", "तिंहों", "इंहों", "दवारा", "इसि", "किंहें", "थि", "उंहों", "ओर", "जिंहें",
    "वहिं", "अभि", "बनि", "हि", "उंहिं", "उंहें", "हें", "वगेरह", "एसे", "रवासा", "कोन",
    "निचे", "काफि", "उसि", "पुरा", "भितर", "हे", "बहि", "वहां", "कोइ", "यहां", "जिंहों",
    "तिंहें", "किसि", "कइ", "यहि", "इंहिं", "जिधर", "इंहें", "अदि", "इतयादि", "हुइ",
    "कोनसा", "इसकि", "दुसरे", "जहां", "अप", "किंहों", "उनकि", "भि", "वरग", "हुअ", "जेसा",
    "नहिं",
];

/// Lucene `IndonesianAnalyzer` default set: `id/stopwords.txt` (355 words).
pub const INDONESIAN: &[&str] = &[
    "ada", "adanya", "adalah", "adapun", "agak", "agaknya", "agar", "akan", "akankah",
    "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar",
    "diantaranya", "antara", "antaranya", "diantara", "apa", "apaan", "mengapa", "apabila",
    "apakah", "apalagi", "apatah", "atau", "ataukah", "ataupun", "bagai", "bagaikan",
    "sebagai", "sebagainya", "bagaimana", "bagaimanapun", "sebagaimana", "bagaimanakah",
    "bagi", "bahkan", "bahwa", "bahwasanya", "sebaliknya", "banyak", "sebanyak", "beberapa",
    "seberapa", "begini", "beginian", "beginikah", "beginilah", "sebegini", "begitu",
    "begitukah", "begitulah", "begitupun", "sebegitu", "belum", "belumlah", "sebelum",
    "sebelumnya", "sebenarnya", "berapa", "berapakah", "berapalah", "berapapun", "betulkah",
    "sebetulnya", "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", "sebisanya",
    "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya",
    "cuma", "percuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "dekat",
    "demi", "demikian", "demikianlah", "sedemikian", "dengan", "depan", "di", "dia",
    "dialah", "dini", "diri", "dirinya", "terdiri", "dong", "dulu", "enggak", "enggaknya",
    "entah", "entahlah", "terhadap", "terhadapnya", "hal", "hampir", "hanya", "hanyalah",
    "harus", "haruslah", "harusnya", "seharusnya", "hendak", "hendaklah", "hendaknya",
    "hingga", "sehingga", "ia", "ialah", "ibarat", "ingin", "inginkah", "inginkan", "ini",
    "inikah", "inilah", "itu", "itukah", "itulah", "jangan", "jangankan", "janganlah",
    "jika", "jikalau", "juga", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian",
    "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun",
    "dikarenakan", "karena", "karenanya", "ke", "kecil", "kemudian", "kenapa", "kepada",
    "kepadanya", "ketika", "seketika", "khususnya", "kini", "kinilah", "kiranya",
    "sekiranya", "kita", "kitalah", "kok", "lagi", "lagian", "selagi", "lah", "lain",
    "lainnya", "melainkan", "selaku", "lalu", "melalui", "terlalu", "lama", "lamanya",
    "selama", "selamanya", "lebih", "terlebih", "bermacam", "macam", "semacam", "maka",
    "makanya", "makin", "malah", "malahan", "mampu", "mampukah", "mana", "manakala",
    "manalagi", "masih", "masihkah", "semasih", "masing", "mau", "maupun", "semaunya",
    "memang", "mereka", "merekalah", "meski", "meskipun", "semula", "mungkin", "mungkinkah",
    "nah", "namun", "nanti", "nantinya", "nyaris", "oleh", "olehnya", "seorang",
    "seseorang", "pada", "padanya", "padahal", "paling", "sepanjang", "pantas",
    "sepantasnya", "sepantasnyalah", "para", "pasti", "pastilah", "per", "pernah", "pula",
    "pun", "merupakan", "rupanya", "serupa", "saat", "saatnya", "sesaat", "saja", "sajalah",
    "saling", "bersama", "sama", "sesama", "sambil", "sampai", "sana", "sangat",
    "sangatlah", "saya", "sayalah", "se", "sebab", "sebabnya", "sebuah", "tersebut",
    "tersebutlah", "sedang", "sedangkan", "sedikit", "sedikitnya", "segala", "segalanya",
    "segera", "sesegera", "sejak", "sejenak", "sekali", "sekalian", "sekalipun", "sesekali",
    "sekaligus", "sekarang", "sekitar", "sekitarnya", "sela", "selain", "selalu", "seluruh",
    "seluruhnya", "semakin", "sementara", "sempat", "semua", "semuanya", "sendiri",
    "sendirinya", "seolah", "seperti", "sepertinya", "sering", "seringnya", "serta",
    "siapa", "siapakah", "siapapun", "disini", "disinilah", "sini", "sinilah", "sesuatu",
    "sesuatunya", "suatu", "sesudah", "sesudahnya", "sudah", "sudahkah", "sudahlah",
    "supaya", "tadi", "tadinya", "tak", "tanpa", "setelah", "telah", "tentang", "tentu",
    "tentulah", "tentunya", "tertentu", "seterusnya", "tapi", "tetapi", "setiap", "tiap",
    "setidaknya", "tidak", "tidakkah", "tidaklah", "toh", "waduh", "wah", "wahai",
    "sewaktu", "walau", "walaupun", "wong", "yaitu", "yakni", "yang",
];

/// Lucene `LatvianAnalyzer` default set: `lv/stopwords.txt` (161 words).
pub const LATVIAN: &[&str] = &[
    "aiz", "ap", "ar", "apakš", "ārpus", "augšpus", "bez", "caur", "dēļ", "gar", "iekš",
    "iz", "kopš", "labad", "lejpus", "līdz", "no", "otrpus", "pa", "par", "pār", "pēc",
    "pie", "pirms", "pret", "priekš", "starp", "šaipus", "uz", "viņpus", "virs", "virspus",
    "zem", "apakšpus", "un", "bet", "jo", "ja", "ka", "lai", "tomēr", "tikko", "turpretī",
    "arī", "kaut", "gan", "tādēļ", "tā", "ne", "tikvien", "vien", "kā", "ir", "te", "vai",
    "kamēr", "diezin", "droši", "diemžēl", "nebūt", "ik", "it", "taču", "nu", "pat",
    "tiklab", "iekšpus", "nedz", "tik", "nevis", "turpretim", "jeb", "iekam", "iekām",
    "iekāms", "kolīdz", "līdzko", "tiklīdz", "jebšu", "tālab", "tāpēc", "nekā", "itin",
    "jā", "jau", "jel", "nē", "nezin", "tad", "tikai", "vis", "tak", "iekams", "būt",
    "biju", "biji", "bija", "bijām", "bijāt", "esmu", "esi", "esam", "esat", "būšu", "būsi",
    "būs", "būsim", "būsiet", "tikt", "tiku", "tiki", "tika", "tikām", "tikāt", "tieku",
    "tiec", "tiek", "tiekam", "tiekat", "tikšu", "tiks", "tiksim", "tiksiet", "tapt",
    "tapi", "tapāt", "topat", "tapšu", "tapsi", "taps", "tapsim", "tapsiet", "kļūt",
    "kļuvu", "kļuvi", "kļuva", "kļuvām", "kļuvāt", "kļūstu", "kļūsti", "kļūst", "kļūstam",
    "kļūstat", "kļūšu", "kļūsi", "kļūs", "kļūsim", "kļūsiet", "varēt", "varēju", "varējām",
    "varēšu", "varēsim", "var", "varēji", "varējāt", "varēsi", "varēsiet", "varat",
    "varēja", "varēs",
];

/// Lucene `LithuanianAnalyzer` default set: `lt/stopwords.txt` (125 words).
pub const LITHUANIAN: &[&str] = &[
    "ant", "apie", "ar", "arba", "aš", "be", "bei", "bet", "bus", "būti", "būtų", "buvo",
    "dėl", "gali", "į", "iki", "ir", "iš", "ja", "ją", "jai", "jais", "jam", "jame", "jas",
    "jei", "ji", "jį", "jie", "jiedu", "jiedvi", "jiedviem", "jiedviese", "jiems", "jis",
    "jo", "jodviem", "jog", "joje", "jomis", "joms", "jos", "jose", "jų", "judu", "judvi",
    "judviejų", "jųdviejų", "judviem", "judviese", "jumis", "jums", "jumyse", "juo",
    "juodu", "juodviese", "juos", "juose", "jus", "jūs", "jūsų", "ką", "kad", "kai", "kaip",
    "kas", "kiek", "kol", "kur", "kurie", "kuris", "man", "mane", "manęs", "manimi", "mano",
    "manyje", "mes", "metu", "mudu", "mudvi", "mudviejų", "mudviem", "mudviese", "mumis",
    "mums", "mumyse", "mus", "mūsų", "nei", "nes", "net", "nors", "nuo", "o", "pat", "per",
    "po", "prie", "prieš", "sau", "save", "savęs", "savimi", "savo", "savyje", "su",
    "tačiau", "tada", "tai", "taip", "tas", "tau", "tave", "tavęs", "tavimi", "tavyje",
    "ten", "to", "todėl", "tu", "tuo", "už", "visi", "yra",
];

/// Lucene `PersianAnalyzer` default set: `fa/stopwords.txt` (308 words).
pub const PERSIAN: &[&str] = &[
    "انان", "نداشته", "سراسر", "خياه", "ايشان", "وي", "تاكنون", "بيشتري", "دوم", "پس",
    "ناشي", "وگو", "يا", "داشتند", "سپس", "هنگام", "هرگز", "پنج", "نشان", "امسال", "ديگر",
    "گروهي", "شدند", "چطور", "ده", "و", "دو", "نخستين", "ولي", "چرا", "چه", "وسط", "ه",
    "كدام", "قابل", "يك", "رفت", "هفت", "همچنين", "در", "هزار", "بله", "بلي", "شايد", "اما",
    "شناسي", "گرفته", "دهد", "داشته", "دانست", "داشتن", "خواهيم", "ميليارد", "وقتيكه",
    "امد", "خواهد", "جز", "اورده", "شده", "بلكه", "خدمات", "شدن", "برخي", "نبود", "بسياري",
    "جلوگيري", "حق", "كردند", "نوعي", "بعري", "نكرده", "نظير", "نبايد", "بوده", "بودن",
    "داد", "اورد", "هست", "جايي", "شود", "دنبال", "داده", "بايد", "سابق", "هيچ", "همان",
    "انجا", "كمتر", "كجاست", "گردد", "كسي", "تر", "مردم", "تان", "دادن", "بودند", "سري",
    "جدا", "ندارند", "مگر", "يكديگر", "دارد", "دهند", "بنابراين", "هنگامي", "سمت", "جا",
    "انچه", "خود", "دادند", "زياد", "دارند", "اثر", "بدون", "بهترين", "بيشتر", "البته",
    "به", "براساس", "بيرون", "كرد", "بعضي", "گرفت", "توي", "اي", "ميليون", "او", "جريان",
    "تول", "بر", "مانند", "برابر", "باشيم", "مدتي", "گويند", "اكنون", "تا", "تنها", "جديد",
    "چند", "بي", "نشده", "كردن", "كردم", "گويد", "كرده", "كنيم", "نمي", "نزد", "روي", "قصد",
    "فقط", "بالاي", "ديگران", "اين", "ديروز", "توسط", "سوم", "ايم", "دانند", "سوي",
    "استفاده", "شما", "كنار", "داريم", "ساخته", "طور", "امده", "رفته", "نخست", "بيست",
    "نزديك", "طي", "كنيد", "از", "انها", "تمامي", "داشت", "يكي", "طريق", "اش", "چيست",
    "روب", "نمايد", "گفت", "چندين", "چيزي", "تواند", "ام", "ايا", "با", "ان", "ايد", "ترين",
    "اينكه", "ديگري", "راه", "هايي", "بروز", "همچنان", "پاعين", "كس", "حدود", "مختلف",
    "مقابل", "چيز", "گيرد", "ندارد", "ضد", "همچون", "سازي", "شان", "مورد", "باره", "مرسي",
    "خويش", "برخوردار", "چون", "خارج", "شش", "هنوز", "تحت", "ضمن", "هستيم", "گفته", "فكر",
    "بسيار", "پيش", "براي", "روزهاي", "انكه", "نخواهد", "بالا", "كل", "وقتي", "كي", "چنين",
    "كه", "گيري", "نيست", "است", "كجا", "كند", "نيز", "يابد", "بندي", "حتي", "توانند",
    "عقب", "خواست", "كنند", "بين", "تمام", "همه", "ما", "باشند", "مثل", "شد", "اري", "باشد",
    "اره", "طبق", "بعد", "اگر", "صورت", "غير", "جاي", "بيش", "ريزي", "اند", "زيرا", "چگونه",
    "بار", "لطفا", "مي", "درباره", "من", "ديده", "همين", "گذاري", "برداري", "علت", "گذاشته",
    "هم", "فوق", "نه", "ها", "شوند", "اباد", "همواره", "هر", "اول", "خواهند", "چهار", "نام",
    "امروز", "مان", "هاي", "قبل", "كنم", "سعي", "تازه", "را", "هستند", "زير", "جلوي",
    "عنوان", "بود",
];

/// Lucene `RomanianAnalyzer` default set: `ro/stopwords.txt` (254 words).
pub const ROMANIAN: &[&str] = &[
    "acea", "aceasta", "această", "aceea", "acei", "aceia", "acel", "acela", "acele",
    "acelea", "acest", "acesta", "aceste", "acestea", "acești", "aceşti", "aceștia",
    "aceştia", "acolo", "acum", "ai", "aia", "aibă", "aici", "al", "ăla", "ale", "alea",
    "ălea", "altceva", "altcineva", "am", "ar", "are", "aș", "aş", "așadar", "aşadar",
    "asemenea", "asta", "ăsta", "astăzi", "astea", "ăstea", "ăștia", "ăştia", "asupra",
    "ați", "aţi", "au", "avea", "avem", "aveți", "aveţi", "azi", "bine", "bucur", "bună",
    "ca", "că", "căci", "când", "care", "cărei", "căror", "cărui", "cât", "câte", "câți",
    "câţi", "către", "câtva", "ce", "cel", "ceva", "chiar", "cînd", "cine", "cineva", "cît",
    "cîte", "cîți", "cîţi", "cîtva", "contra", "cu", "cum", "cumva", "curând", "curînd",
    "da", "dă", "dacă", "dar", "datorită", "de", "deci", "deja", "deoarece", "departe",
    "deși", "deşi", "din", "dinaintea", "dintr", "dintre", "drept", "după", "ea", "ei",
    "el", "ele", "eram", "este", "ești", "eşti", "eu", "face", "fără", "fi", "fie",
    "fiecare", "fii", "fim", "fiți", "fiţi", "iar", "ieri", "îi", "îl", "îmi", "împotriva",
    "în", "înainte", "înaintea", "încât", "încît", "încotro", "între", "întrucât",
    "întrucît", "îți", "îţi", "la", "lângă", "le", "li", "lîngă", "lor", "lui", "mă",
    "mâine", "mea", "mei", "mele", "mereu", "meu", "mi", "mine", "mult", "multă", "mulți",
    "mulţi", "ne", "nicăieri", "nici", "nimeni", "niște", "nişte", "noastră", "noastre",
    "noi", "noștri", "noştri", "nostru", "nu", "ori", "oricând", "oricare", "oricât",
    "orice", "oricînd", "oricine", "oricît", "oricum", "oriunde", "până", "pe", "pentru",
    "peste", "pînă", "poate", "pot", "prea", "prima", "primul", "prin", "printr", "sa",
    "să", "săi", "sale", "sau", "său", "se", "și", "şi", "sînt", "sîntem", "sînteți",
    "sînteţi", "spre", "sub", "sunt", "suntem", "sunteți", "sunteţi", "ta", "tăi", "tale",
    "tău", "te", "ți", "ţi", "ție", "ţie", "tine", "toată", "toate", "tot", "toți", "toţi",
    "totuși", "totuşi", "tu", "un", "una", "unde", "undeva", "unei", "unele", "uneori",
    "unor", "vă", "vi", "voastră", "voastre", "voi", "voștri", "voştri", "vostru", "vouă",
    "vreo", "vreun",
];

/// Lucene `SerbianAnalyzer` default set: `sr/stopwords.txt` (156 words).
pub const SERBIAN: &[&str] = &[
    "i", "ili", "a", "ali", "pa", "biti", "ne", "jesam", "sam", "jesi", "si", "je", "jesmo",
    "smo", "jeste", "ste", "jesu", "su", "nijesam", "nisam", "nijesi", "nisi", "nije",
    "nijesmo", "nismo", "nijeste", "niste", "nijesu", "nisu", "budem", "budeš", "bude",
    "budemo", "budete", "budu", "budes", "bih", "bi", "bismo", "biste", "biše", "bise",
    "bio", "bili", "budimo", "budite", "bila", "bilo", "bile", "ću", "ćeš", "će", "ćemo",
    "ćete", "neću", "nećeš", "neće", "nećemo", "nećete", "cu", "ces", "ce", "cemo", "cete",
    "necu", "neces", "nece", "necemo", "necete", "mogu", "možeš", "može", "možemo",
    "možete", "mozes", "moze", "mozemo", "mozete", "и", "или", "а", "али", "па", "бити",
    "не", "јесам", "сам", "јеси", "си", "је", "јесмо", "смо", "јесте", "сте", "јесу", "су",
    "нијесам", "нисам", "нијеси", "ниси", "није", "нијесмо", "нисмо", "нијесте", "нисте",
    "нијесу", "нису", "будем", "будеш", "буде", "будемо", "будете", "буду", "будес", "бих",
    "би", "бисмо", "бисте", "бише", "бисе", "био", "били", "будимо", "будите", "била",
    "било", "биле", "ћу", "ћеш", "ће", "ћемо", "ћете", "нећу", "нећеш", "неће", "нећемо",
    "нећете", "цу", "цес", "це", "цемо", "цете", "нецу", "нецес", "неце", "нецемо",
    "нецете", "могу", "можеш", "може", "можемо", "можете", "мозес", "мозе", "моземо",
    "мозете",
];

/// Lucene `SoraniAnalyzer` default set: `ckb/stopwords.txt` (62 words).
pub const SORANI: &[&str] = &[
    "و", "کە", "ی", "کرد", "ئەوەی", "سەر", "دوو", "هەروەها", "لەو", "دەکات", "چەند", "هەر",
    "ئەو", "ئەم", "من", "ئێمە", "تۆ", "ئێوە", "ئەوان", "بە", "پێ", "بەبێ", "بەدەم", "بەلای",
    "بەپێی", "بەرلە", "بەرەوی", "بەرەوە", "بەردەم", "بێ", "بێجگە", "بۆ", "دە", "تێ",
    "دەگەڵ", "دوای", "جگە", "لە", "لێ", "لەبەر", "لەبەینی", "لەبابەت", "لەبارەی", "لەباتی",
    "لەبن", "لەبرێتی", "لەدەم", "لەگەڵ", "لەلایەن", "لەناو", "لەنێو", "لەپێناوی", "لەرەوی",
    "لەرێ", "لەرێگا", "لەسەر", "لەژێر", "ناو", "نێوان", "پاش", "پێش", "وەک",
];

/// Lucene `ThaiAnalyzer` default set: `th/stopwords.txt` (87 words).
pub const THAI: &[&str] = &[
    "ไว้", "ไม่", "ให้", "ใน", "โดย", "ได้", "แห่ง", "แล้ว", "และ", "แต่", "เอง", "เลย",
    "เรา", "เมื่อ", "เพื่อ", "เพราะ", "เป็นการ", "เป็น", "เนื่องจาก", "เดียวกัน", "เดียว",
    "เช่น", "เฉพาะ", "เคย", "เขา", "อีก", "อาจ", "อะไร", "อย่าง", "อยู่", "อยาก", "หาก",
    "หลาย", "หลังจาก", "หลัง", "หรือ", "หนึ่ง", "ส่วน", "สุด", "สําหรับ", "สำหรับ", "ว่า",
    "ระหว่าง", "รวม", "ยัง", "มี", "มาก", "พร้อม", "บาง", "น่า", "นี้", "นั้น", "นอกจาก",
    "ทุก", "ที่สุด", "ที่", "ทําให้", "ทำให้", "ทั้งนี้", "ทั้ง", "ถ้า", "ถูก", "ถึง",
    "ต้อง", "ต่างๆ", "ต่าง", "ต่อ", "ตาม", "ตั้งแต่", "ด้วย", "ดัง", "ซึ่ง", "จึง", "จาก",
    "จะ", "คือ", "ความ", "ครั้ง", "คง", "ของ", "ขณะ", "ก่อน", "ก็", "การ", "กับ", "กัน",
    "กว่า",
];

/// Lucene `TurkishAnalyzer` default set: `tr/stopwords.txt` (209 words).
pub const TURKISH: &[&str] = &[
    "acaba", "altmış", "altı", "ama", "ancak", "arada", "aslında", "ayrıca", "bana", "bazı",
    "belki", "ben", "benden", "beni", "benim", "beri", "beş", "bile", "bin", "bir",
    "birçok", "biri", "birkaç", "birkez", "birşey", "birşeyi", "biz", "bize", "bizden",
    "bizi", "bizim", "böyle", "böylece", "bu", "buna", "bunda", "bundan", "bunlar",
    "bunları", "bunların", "bunu", "bunun", "burada", "çok", "çünkü", "da", "daha", "dahi",
    "de", "defa", "değil", "diğer", "diye", "doksan", "dokuz", "dolayı", "dolayısıyla",
    "dört", "edecek", "eden", "ederek", "edilecek", "ediliyor", "edilmesi", "ediyor",
    "eğer", "elli", "en", "etmesi", "etti", "ettiği", "ettiğini", "gibi", "göre", "halen",
    "hangi", "hatta", "hem", "henüz", "hep", "hepsi", "her", "herhangi", "herkesin", "hiç",
    "hiçbir", "için", "iki", "ile", "ilgili", "ise", "işte", "itibaren", "itibariyle",
    "kadar", "karşın", "katrilyon", "kendi", "kendilerine", "kendini", "kendisi",
    "kendisine", "kendisini", "kez", "ki", "kim", "kimden", "kime", "kimi", "kimse", "kırk",
    "milyar", "milyon", "mu", "mü", "mı", "nasıl", "ne", "neden", "nedenle", "nerde",
    "nerede", "nereye", "niye", "niçin", "o", "olan", "olarak", "oldu", "olduğu",
    "olduğunu", "olduklarını", "olmadı", "olmadığı", "olmak", "olması", "olmayan", "olmaz",
    "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "on", "ona", "ondan", "onlar",
    "onlardan", "onları", "onların", "onu", "onun", "otuz", "oysa", "öyle", "pek", "rağmen",
    "sadece", "sanki", "sekiz", "seksen", "sen", "senden", "seni", "senin", "siz", "sizden",
    "sizi", "sizin", "şey", "şeyden", "şeyi", "şeyler", "şöyle", "şu", "şuna", "şunda",
    "şundan", "şunları", "şunu", "tarafından", "trilyon", "tüm", "üç", "üzere", "var",
    "vardı", "ve", "veya", "ya", "yani", "yapacak", "yapılan", "yapılması", "yapıyor",
    "yapmak", "yaptı", "yaptığı", "yaptığını", "yaptıkları", "yedi", "yerine", "yetmiş",
    "yine", "yirmi", "yoksa", "yüz", "zaten",
];

/// Every language name accepted by [`for_language`] (bare form, without
/// ES's surrounding underscores).
pub const LANGUAGES: &[&str] = &[
    "english", "danish", "dutch", "finnish", "french", "german", "hungarian", "italian",
    "norwegian", "portuguese", "russian", "spanish", "swedish", "irish", "arabic",
    "armenian", "basque", "bengali", "brazilian", "bulgarian", "catalan", "cjk", "czech",
    "estonian", "galician", "greek", "hindi", "indonesian", "latvian", "lithuanian",
    "persian", "romanian", "serbian", "sorani", "thai", "turkish",
];

/// Look up a built-in stop word list by ES language name (e.g. "french",
/// "_french_", "brazilian", "cjk").
///
/// Accepts both the bare name and ES's underscore form, case-insensitively.
/// `"_none_"` (or `"none"`) yields an empty list; unknown names yield `None`.
pub fn for_language(name: &str) -> Option<&'static [&'static str]> {
    let trimmed = name.trim();
    let bare = trimmed
        .strip_prefix('_')
        .and_then(|s| s.strip_suffix('_'))
        .unwrap_or(trimmed);
    let lower = bare.to_ascii_lowercase();
    match lower.as_str() {
        "none" => Some(&[]),
        "english" => Some(ENGLISH),
        "danish" => Some(DANISH),
        "dutch" => Some(DUTCH),
        "finnish" => Some(FINNISH),
        "french" => Some(FRENCH),
        "german" => Some(GERMAN),
        "hungarian" => Some(HUNGARIAN),
        "italian" => Some(ITALIAN),
        "norwegian" => Some(NORWEGIAN),
        "portuguese" => Some(PORTUGUESE),
        "russian" => Some(RUSSIAN),
        "spanish" => Some(SPANISH),
        "swedish" => Some(SWEDISH),
        "irish" => Some(IRISH),
        "arabic" => Some(ARABIC),
        "armenian" => Some(ARMENIAN),
        "basque" => Some(BASQUE),
        "bengali" => Some(BENGALI),
        "brazilian" => Some(BRAZILIAN),
        "bulgarian" => Some(BULGARIAN),
        "catalan" => Some(CATALAN),
        "cjk" => Some(CJK),
        "czech" => Some(CZECH),
        "estonian" => Some(ESTONIAN),
        "galician" => Some(GALICIAN),
        "greek" => Some(GREEK),
        "hindi" => Some(HINDI),
        "indonesian" => Some(INDONESIAN),
        "latvian" => Some(LATVIAN),
        "lithuanian" => Some(LITHUANIAN),
        "persian" => Some(PERSIAN),
        "romanian" => Some(ROMANIAN),
        "serbian" => Some(SERBIAN),
        "sorani" => Some(SORANI),
        "thai" => Some(THAI),
        "turkish" => Some(TURKISH),
        _ => None,
    }
}

/// `(language, source file under testdata/stopwords, snowball format)` for
/// every list generated from a Lucene resource file.
#[cfg(test)]
pub(crate) const SOURCES: &[(&str, &str, bool)] = &[
    ("danish", "danish_stop.txt", true),
    ("dutch", "dutch_stop.txt", true),
    ("finnish", "finnish_stop.txt", true),
    ("french", "french_stop.txt", true),
    ("german", "german_stop.txt", true),
    ("hungarian", "hungarian_stop.txt", true),
    ("italian", "italian_stop.txt", true),
    ("norwegian", "norwegian_stop.txt", true),
    ("portuguese", "portuguese_stop.txt", true),
    ("russian", "russian_stop.txt", true),
    ("spanish", "spanish_stop.txt", true),
    ("swedish", "swedish_stop.txt", true),
    ("irish", "irish_stop.txt", true),
    ("arabic", "ar_stopwords.txt", false),
    ("armenian", "hy_stopwords.txt", false),
    ("basque", "eu_stopwords.txt", false),
    ("bengali", "bn_stopwords.txt", false),
    ("brazilian", "br_stopwords.txt", false),
    ("bulgarian", "bg_stopwords.txt", false),
    ("catalan", "ca_stopwords.txt", false),
    ("cjk", "cjk_stopwords.txt", false),
    ("czech", "cz_stopwords.txt", false),
    ("estonian", "et_stopwords.txt", false),
    ("galician", "gl_stopwords.txt", false),
    ("greek", "el_stopwords.txt", false),
    ("hindi", "hi_stopwords.txt", false),
    ("indonesian", "id_stopwords.txt", false),
    ("latvian", "lv_stopwords.txt", false),
    ("lithuanian", "lt_stopwords.txt", false),
    ("persian", "fa_stopwords.txt", false),
    ("romanian", "ro_stopwords.txt", false),
    ("serbian", "sr_stopwords.txt", false),
    ("sorani", "ckb_stopwords.txt", false),
    ("thai", "th_stopwords.txt", false),
    ("turkish", "tr_stopwords.txt", false),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn english_has_33_unique_words() {
        assert_eq!(ENGLISH.len(), 33);
        let set: HashSet<&str> = ENGLISH.iter().copied().collect();
        assert_eq!(set.len(), 33);
    }

    #[test]
    fn english_contains_expected_words() {
        for w in ["the", "and", "with", "is", "not"] {
            assert!(ENGLISH.contains(&w), "missing {w}");
        }
    }

    #[test]
    fn every_list_is_nonempty_lowercase_and_unique() {
        for lang in LANGUAGES {
            let list = for_language(lang).unwrap_or_else(|| panic!("{lang} not resolvable"));
            assert!(!list.is_empty(), "{lang} is empty");
            let mut seen = HashSet::new();
            for w in list {
                assert!(!w.is_empty(), "{lang} has an empty entry");
                assert!(!w.chars().any(char::is_whitespace), "{lang}: {w:?} has whitespace");
                assert_eq!(*w, w.to_lowercase(), "{lang}: {w:?} is not lowercase");
                assert!(seen.insert(*w), "{lang}: duplicate {w:?}");
            }
        }
    }

    #[test]
    fn languages_list_is_unique_and_round_trips() {
        let set: HashSet<&str> = LANGUAGES.iter().copied().collect();
        assert_eq!(set.len(), LANGUAGES.len());
        assert!(LANGUAGES.contains(&"english"));
        assert!(LANGUAGES.contains(&"cjk"));
        for lang in LANGUAGES {
            assert!(for_language(lang).is_some(), "{lang}");
            assert!(for_language(&format!("_{lang}_")).is_some(), "_{lang}_");
        }
    }

    #[test]
    fn for_language_handles_underscore_form_case_and_unknown() {
        let french = for_language("french").unwrap();
        assert_eq!(for_language("_french_").unwrap(), french);
        assert_eq!(for_language("FRENCH").unwrap(), french);
        assert_eq!(for_language("_French_").unwrap(), french);
        assert_eq!(for_language("_english_").unwrap(), ENGLISH);
        assert_eq!(for_language("english").unwrap(), ENGLISH);
        assert_eq!(for_language("_none_"), Some(&[][..]));
        assert_eq!(for_language("klingon"), None);
        assert_eq!(for_language(""), None);
        assert_eq!(for_language("__"), None);
    }

    #[test]
    fn spot_check_known_words() {
        let cases: &[(&str, &[&str])] = &[
            ("french", &["le", "la", "les", "de", "et"]),
            ("german", &["der", "die", "das", "und", "nicht"]),
            ("spanish", &["el", "la", "de", "que", "y"]),
            ("italian", &["il", "la", "di", "che", "e"]),
            ("portuguese", &["de", "a", "o", "que", "não"]),
            ("dutch", &["de", "en", "van", "het", "een"]),
            ("swedish", &["och", "det", "att", "i", "en"]),
            ("danish", &["og", "i", "det", "at", "en"]),
            ("norwegian", &["og", "i", "det", "at", "en"]),
            ("finnish", &["ja", "on", "ei", "että", "se"]),
            ("hungarian", &["a", "az", "és", "hogy", "nem"]),
            ("russian", &["и", "в", "не", "что", "на"]),
            ("arabic", &["من", "في", "على", "عن", "ما"]),
            ("turkish", &["ve", "bir", "bu", "için", "de"]),
            ("czech", &["a", "je", "se", "na", "to"]),
            ("greek", &["και", "το", "του", "με", "να"]),
            ("indonesian", &["yang", "dan", "di", "dari", "dengan"]),
            // Note: "untuk" is NOT in Lucene's indonesian_stopwords.txt.
            ("persian", &["و", "در", "از", "كه", "با"]),
            ("hindi", &["और", "का", "के", "की", "है"]),
            ("thai", &["และ", "ที่", "ใน", "ของ", "เป็น"]),
            ("romanian", &["şi", "de", "la", "în", "cu"]),
            ("bulgarian", &["и", "в", "на", "не", "се"]),
            ("catalan", &["el", "la", "de", "que", "i"]),
            ("brazilian", &["de", "a", "o", "que", "nao"]),
            ("cjk", &["the", "www", "s", "t"]),
            ("irish", &["agus", "an", "na", "ar", "is"]),
            ("basque", &["eta", "da", "ez", "bat", "du"]),
            ("galician", &["e", "o", "a", "de", "que"]),
            ("estonian", &["ja", "ei", "et", "see", "mis"]),
            ("latvian", &["un", "ir", "ar", "no", "par"]),
            ("lithuanian", &["ir", "yra", "kad", "su", "iš"]),
            ("armenian", &["և", "որ", "է", "այս", "մի"]),
            ("serbian", &["i", "ili", "a", "ali", "pa"]),
        ];
        for (lang, words) in cases {
            let list = for_language(lang).unwrap();
            for w in *words {
                assert!(list.contains(w), "{lang} should contain {w:?}");
            }
        }
    }

    /// Lucene `WordlistLoader.getSnowballWordSet`: drop text after `|`,
    /// split on whitespace.
    fn parse_snowball(src: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for line in src.lines() {
            let line = line.split('|').next().unwrap_or("");
            for w in line.split_whitespace() {
                if !out.iter().any(|x| x == w) {
                    out.push(w.to_string());
                }
            }
        }
        out
    }

    /// Lucene `WordlistLoader.getWordSet(reader, "#")`: skip `#` lines,
    /// trim, skip blanks.
    fn parse_plain(src: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for line in src.lines() {
            if line.starts_with('#') {
                continue;
            }
            let w = line.trim();
            if !w.is_empty() && !out.iter().any(|x| x == w) {
                out.push(w.to_string());
            }
        }
        out
    }

    fn source_text(file: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/analysis/testdata/stopwords")
            .join(file);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    #[test]
    fn every_list_matches_its_lucene_source_file() {
        assert_eq!(SOURCES.len() + 1, LANGUAGES.len(), "every non-English list has a source");
        for (lang, file, snowball) in SOURCES {
            let src = source_text(file);
            let expected = if *snowball { parse_snowball(&src) } else { parse_plain(&src) };
            let actual = for_language(lang).unwrap();
            assert_eq!(actual.len(), expected.len(), "{lang}: entry count vs {file}");
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert_eq!(a, e, "{lang}: entry mismatch vs {file}");
            }
        }
    }

    #[test]
    fn list_sizes_match_lucene() {
        let sizes: &[(&str, usize)] = &[
            ("danish", 94), ("dutch", 101), ("finnish", 229), ("french", 154),
            ("german", 231), ("hungarian", 198), ("italian", 279), ("norwegian", 172),
            ("portuguese", 203), ("russian", 159), ("spanish", 308), ("swedish", 114),
            ("irish", 109), ("arabic", 119), ("armenian", 45), ("basque", 98),
            ("bengali", 116), ("brazilian", 127), ("bulgarian", 190), ("catalan", 218),
            ("cjk", 35), ("czech", 171), ("estonian", 1470), ("galician", 160),
            ("greek", 75), ("hindi", 225), ("indonesian", 355), ("latvian", 161),
            ("lithuanian", 125), ("persian", 308), ("romanian", 254), ("serbian", 156),
            ("sorani", 62), ("thai", 87), ("turkish", 209), ("english", 33),
        ];
        assert_eq!(sizes.len(), LANGUAGES.len());
        for (lang, n) in sizes {
            assert_eq!(for_language(lang).unwrap().len(), *n, "{lang}");
        }
    }
}
