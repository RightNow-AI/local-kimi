"""Deterministic prompt and teacher-forced corpora for the accuracy lane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AccuracyPrompt:
    prompt_id: str
    category: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


_FACTUAL = (
    "What is the capital of Morocco? Answer with the city only.",
    "Which chemical element has atomic number 26? Answer with its name.",
    "Who wrote the novel Frankenstein?",
    "How many degrees are in the interior angles of a triangle?",
    "What is the largest planet in the Solar System?",
    "Which ocean lies between Africa and Australia?",
    "What is the SI unit of electric current?",
    "In what year did the Berlin Wall fall?",
    "What gas do plants absorb during photosynthesis?",
    "Which organ pumps blood through the human body?",
    "What is the square root of 144?",
    "Which language is primarily spoken in Brazil?",
    "What is the freezing point of pure water in Celsius at standard pressure?",
    "Which continent contains the Sahara Desert?",
    "Who painted The Starry Night?",
    "What is the smallest prime number?",
    "Which planet is known for its prominent ring system?",
    "What metal is liquid near ordinary room temperature?",
    "What is the currency of Japan?",
    "Which blood cells primarily carry oxygen?",
    "What is the longest river in South America?",
    "Which instrument measures atmospheric pressure?",
    "What is the binary representation of decimal 10?",
    "Which layer of Earth is directly beneath the crust?",
    "What is the main language of the source code in the CPython interpreter?",
    "Which theorem relates the sides of a right triangle?",
)

_REASONING = (
    "A train travels 180 kilometers in 3 hours, then 140 kilometers in 2 hours. What is its average speed for the whole trip? Show the calculation.",
    "Mira has twice as many blue marbles as red marbles and 7 green marbles. She has 31 marbles total. How many red marbles does she have?",
    "All kestrels are birds. No birds in the reserve are mammals. Can any kestrel in the reserve be a mammal? Explain briefly.",
    "A shop discounts a 240 dollar item by 15 percent and then adds 8 percent sales tax. What is the final price?",
    "The sequence begins 2, 6, 12, 20, 30. State the next term and the rule.",
    "Three boxes are labeled Apples, Oranges, and Mixed, but every label is wrong. You may draw one fruit from one box. Which box should you draw from first, and why?",
    "A rectangle has perimeter 54 and length 17. What is its area?",
    "If five machines make five parts in five minutes at identical rates, how many parts do twenty machines make in twenty minutes?",
    "Nora is older than Sam. Sam is older than Ilya. Ilya is older than Chen. List them from oldest to youngest.",
    "A fair coin is tossed three times. What is the probability of exactly two heads?",
    "A tank is one quarter full. Adding 45 liters makes it five eighths full. What is the tank capacity?",
    "Find the smallest positive integer divisible by 6, 8, and 15.",
    "A clock shows 3:00. What is the smaller angle between its hands at 3:30?",
    "A farmer has chickens and goats. There are 18 heads and 50 legs. How many goats are there?",
    "A password has two distinct letters followed by two distinct digits. How many passwords are possible if order matters?",
    "A book has pages numbered 1 through 250. How many times is the digit 2 printed?",
    "An urn contains 4 red, 5 blue, and 3 green balls. Two are drawn without replacement. What is the probability both are blue?",
    "The mean of six numbers is 14. Five of the numbers sum to 61. What is the sixth number?",
    "A runner completes the first half of a route at 8 km/h and the second half at 12 km/h. What is the average speed for the route?",
    "If x + y = 14 and xy = 40, what are the possible values of x and y?",
    "A meeting starts at 22:45 UTC and lasts 2 hours 35 minutes. At what UTC time does it end?",
    "Four people each shake hands with every other person exactly once. How many handshakes occur?",
    "A cube has total surface area 150 square centimeters. What is its volume?",
    "You have 9 identical-looking coins and one is heavier. What is the fewest balance-scale weighings needed to guarantee finding it? Explain.",
    "A number increases by 20 percent and then decreases by 20 percent. Is the final number equal to the original? Quantify the change.",
    "A path graph has 12 vertices. How many edges does it have, and why?",
)

_CODE = (
    "Write a Python function that returns the first non-repeating character in a string, or None if there is none. Include time complexity.",
    "Write a JavaScript function that groups an array of objects by a specified key without mutating the input.",
    "Give a SQL query that returns each customer and their most recent order, including customers with no orders.",
    "Implement binary search in Rust for a sorted slice of i64 values and return Option<usize>.",
    "Write a Python generator that yields fixed-size chunks from any iterable without loading the entire iterable.",
    "Show a TypeScript discriminated union for success and error API results, plus an exhaustive handler.",
    "Write a Bash command that finds regular .log files older than seven days under /var/tmp and prints their paths without deleting them.",
    "Implement a Go function that reverses a UTF-8 string by Unicode code points.",
    "Write a Python context manager that measures elapsed wall-clock time and exposes the result after exit.",
    "Give a PostgreSQL transaction that transfers an amount between two account rows while preventing a negative balance.",
    "Write a Java method that merges two sorted integer arrays into one sorted array in linear time.",
    "Implement an LRU cache in Python using OrderedDict with get and put operations.",
    "Write a C function that safely computes the length of a null-terminated string with a maximum scan bound.",
    "Show a React hook that debounces a changing value and cleans up timers correctly.",
    "Write a regular expression and a short Python example that validates lowercase hexadecimal strings of exactly 64 characters.",
    "Implement breadth-first traversal of an adjacency-list graph in Python and return visit order.",
    "Write a Kotlin function that partitions a list into values satisfying a predicate and values that do not.",
    "Give a Dockerfile snippet that runs a Python service as a non-root user and uses a pinned base-image digest placeholder.",
    "Write a Python function that atomically replaces a JSON file by writing and fsyncing a temporary sibling first.",
    "Implement a min-heap push operation from scratch in pseudocode and state its complexity.",
    "Write a C# async method that fetches three independent URLs concurrently and preserves input order in the results.",
    "Give a SQLite query that finds duplicate email addresses after trimming whitespace and lowercasing.",
    "Write a Python function that compares semantic version triples without using a third-party package.",
    "Implement matrix transpose for a rectangular list of lists in Python and reject ragged input.",
    "Show how to parse newline-delimited JSON incrementally in Node.js without reading the full file into memory.",
    "Write a property-based test idea for a function that encodes and decodes signed four-bit integers packed two per byte.",
)

_INSTRUCTION = (
    "Return exactly three bullet points explaining photosynthesis. Each bullet must contain exactly eight words.",
    "Summarize the benefits and risks of caching in a two-column Markdown table with four data rows.",
    "Explain recursion without using the words function, call, stack, or itself.",
    "Respond with valid JSON containing keys answer, confidence, and assumptions for the question: Is 97 prime?",
    "Rewrite 'The deployment failed because the token expired' in a neutral incident-report style, exactly one sentence.",
    "List five planets in order from the Sun, but omit Earth and number the list starting at 0.",
    "Give directions for making tea in exactly four imperative sentences, with no introductory text.",
    "Classify the sentiment of 'The update fixed my problem but made startup slower' as JSON with positive and negative evidence arrays.",
    "Describe a bicycle using one simile, one semicolon, and no more than 35 words.",
    "Translate 'Good morning, where is the train station?' into Spanish and French, one labeled line per language.",
    "Provide a regex for an ISO date YYYY-MM-DD, then give one matching and one non-matching example. Use exactly three lines.",
    "Answer whether a whale is a fish using the structure Claim, Evidence, Conclusion, each on its own line.",
    "Create a six-item checklist for reviewing an API change. Every item must begin with a verb.",
    "Explain the difference between encryption and hashing in exactly 50 words.",
    "Return a CSV with header name,kind and rows for oak, salmon, granite, and oxygen.",
    "Write a polite refusal to share a password in fewer than 20 words and offer one safe alternative.",
    "Convert the sentence 'Logs reveal failures quickly' into passive voice, then into a question, with labels.",
    "Name four sorting algorithms ordered by typical average-case complexity, and include Big O notation.",
    "Explain a database index to a twelve-year-old in two short paragraphs and use a library analogy.",
    "Return only the uppercase initials of 'central processing unit'.",
    "Give one argument for and one against remote work. Each must be a single sentence beginning with For: or Against:.",
    "Produce a YAML object with keys service, replicas, and ports, using service value api and two numeric ports.",
    "Describe the water cycle in chronological order using exactly five numbered steps.",
    "Correct the grammar in 'Neither of the servers are responding' and briefly state the rule in parentheses.",
    "Answer the question 'What is 19 squared?' with only the integer and no punctuation.",
    "Provide three alternative titles for an article about reliable backups. Each title must be under six words.",
)

_CITIES = (
    "Amman",
    "Lima",
    "Oslo",
    "Dakar",
    "Kyoto",
    "Accra",
    "Riga",
    "Quito",
)
_COLORS = ("amber", "blue", "coral", "green", "indigo", "silver", "violet")
_OBJECTS = ("compass", "lantern", "notebook", "telescope", "vase", "whistle")


def _long_context_prompts() -> tuple[str, ...]:
    prompts = []
    for prompt_index in range(26):
        records = []
        for record_index in range(96):
            record_id = f"L{prompt_index:02d}-{record_index:03d}"
            city = _CITIES[(prompt_index + 3 * record_index) % len(_CITIES)]
            color = _COLORS[(2 * prompt_index + record_index) % len(_COLORS)]
            object_name = _OBJECTS[(prompt_index + 5 * record_index) % len(_OBJECTS)]
            value = 100 + ((prompt_index + 11) * (record_index + 7)) % 900
            records.append(
                f"Record {record_id}: city={city}; color={color}; "
                f"object={object_name}; value={value}."
            )
        target_index = (prompt_index * 37 + 11) % len(records)
        target_id = f"L{prompt_index:02d}-{target_index:03d}"
        prompts.append(
            "Use only the reference ledger below. Ignore patterns in the IDs and "
            "retrieve the requested row exactly.\n\n"
            + "\n".join(records)
            + f"\n\nQuestion: For record {target_id}, return city, color, object, and value "
            "in that order, separated by vertical bars."
        )
    return tuple(prompts)


def build_prompt_set() -> tuple[AccuracyPrompt, ...]:
    groups = (
        ("factual", _FACTUAL),
        ("reasoning", _REASONING),
        ("code", _CODE),
        ("instruction", _INSTRUCTION),
        ("long_context_recall", _long_context_prompts()),
    )
    prompts = []
    for category, texts in groups:
        for index, prompt in enumerate(texts):
            prompts.append(
                AccuracyPrompt(
                    prompt_id=f"{category}-{index:03d}",
                    category=category,
                    text=prompt,
                )
            )
    if len(prompts) != 130:
        raise AssertionError(f"the accuracy prompt set must contain 130 prompts, got {len(prompts)}")
    return tuple(prompts)


def prompt_set_sha256(prompts: tuple[AccuracyPrompt, ...] | None = None) -> str:
    selected = build_prompt_set() if prompts is None else prompts
    payload = json.dumps(
        [prompt.as_dict() for prompt in selected],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_TEACHER_PARAGRAPHS = (
    "A reliable measurement begins by naming the variable that is allowed to change. If hardware, runtime, prompts, decoding rules, and model structure remain fixed, a difference in output can be assigned to the changed weights. When several factors move together, a precise number can still be produced, but the number no longer answers a precise causal question. Controlled comparisons are therefore less about collecting many figures than about preventing alternative explanations from entering the result.",
    "During a winter survey, field researchers placed temperature sensors at the river bank, beneath the tree canopy, and on an exposed ridge. The ridge warmed fastest after sunrise but also cooled fastest after sunset. The canopy changed slowly because leaves reduced direct radiation and trapped a shallow layer of air. The bank followed the water temperature and showed the smallest daily swing. These patterns helped the team distinguish local weather from instrument faults.",
    "A database transaction transfers value safely only when all related updates share one atomic boundary. The debit should not become visible without the credit, and concurrent writers should not both spend the same balance. Row locks, a checked balance condition, and a rollback on failure form a simple protocol. The exact syntax varies between systems, but the invariant does not: observers must see either the complete transfer or no transfer at all.",
    "The museum catalog described a brass navigation instrument found in a cedar case. Its outer ring carried degree marks, while a movable arm aligned with a sighting vane. Salt residue suggested use near the coast. Conservators photographed every surface before cleaning, recorded the order of removed screws, and stored loose fibers separately. That sequence preserved evidence about construction and use before restoration changed the object.",
    "Compilers transform source programs through several representations. A parser first turns tokens into a structured tree. Semantic analysis resolves names and checks types. Later passes lower high-level operations into simpler instructions, optimize repeated work, and choose machine operations for a target processor. Each pass should preserve the observable behavior of the program, although debugging information and performance can change. Tests often compare both final results and intermediate invariants.",
    "A community garden divided irrigation into short morning cycles. Sandy beds received smaller, more frequent doses because water drained quickly, while clay beds received slower doses with longer pauses to prevent runoff. Volunteers measured moisture below the surface instead of judging only by leaf appearance. After two weeks they reduced total water use without lowering growth, showing that timing and soil structure mattered as much as the daily volume.",
    "When a message crosses an unreliable network, acknowledgements and stable identifiers help separate retries from new work. A sender may repeat a request after a timeout even when the first attempt succeeded. If the receiver records the identifier and original result, the retry can return that result without applying the operation twice. This is especially important for payments, reservations, and any action whose duplicate cannot be repaired by simply sending another message.",
    "The astronomy class compared spectra from three stars. One spectrum peaked toward blue wavelengths, one near yellow, and one toward red. Dark absorption lines appeared at matching wavelengths even though their strengths differed. The students inferred that temperature explained the broad color shift while chemical elements explained the shared line positions. They also noted that motion could displace every line together, providing a separate measure of radial velocity.",
    "An emergency checklist should be short enough to use under pressure and specific enough to prevent omission. Operators first confirm personal safety, then identify the affected system, preserve evidence, limit further damage, and establish a communication channel. Recovery begins only after the scope is understood. A checklist cannot replace judgment, but it can reserve attention for judgment by making routine sequencing explicit.",
    "A baker testing flour blends kept oven temperature, loaf mass, mixing time, and proofing time constant. Only the fraction of whole-grain flour changed. Higher fractions absorbed more water and produced a tighter crumb until hydration was adjusted in a later experiment. Because the first trial changed one factor, it established the direction of the effect. The follow-up then tested whether added water could recover texture without erasing the flavor difference.",
    "Versioned file formats need rules for both readers and writers. A writer should declare the format it emits, while a reader should reject unknown required fields rather than silently guessing. Optional fields can extend the format when their absence has a defined meaning. Migration tools should preserve information they do not understand or fail visibly. These practices turn compatibility from an informal hope into a contract that can be tested.",
    "The coastal train left before dawn and crossed three distinct landscapes. It began among warehouses beside the harbor, climbed through olive groves, and entered a high plateau covered in pale grass. At each station the conductor compared a paper manifest with tagged crates in the baggage car. One crate remained sealed because its destination was the final stop. The repeated checks made the chain of custody visible without opening the cargo.",
)


def teacher_forced_text() -> str:
    sections = []
    for cycle in range(3):
        for index, paragraph in enumerate(_TEACHER_PARAGRAPHS):
            sections.append(f"Section {cycle + 1}.{index + 1}. {paragraph}")
    return "\n\n".join(sections)


def teacher_text_sha256() -> str:
    return hashlib.sha256(teacher_forced_text().encode("utf-8")).hexdigest()
