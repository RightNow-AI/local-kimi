# Licence position for the Kimi-Linear base model

This records a decision, who made it, and the evidence it rests on, so a reader
can evaluate it rather than take it on trust.

## The situation

`moonshotai/Kimi-Linear-48B-A3B-Instruct` on HuggingFace declares `license: mit`
in its model-card front matter and ships **no licence instrument**. Verified
twice and independently:

- The HuggingFace API file listing for the repo contains no `LICENSE`,
  `LICENSE.md`, `LICENSE.txt` or `NOTICE`.
- A full snapshot of the repository was downloaded to a Modal volume during this
  work and the resulting file census reported `license_files: []`.

This is not specific to one repo. The same pattern holds for
`Kimi-Linear-48B-A3B-Base`, `Kimi-VL-A3B-Instruct` and
`Moonlight-16B-A3B-Instruct`. `moonshotai/Kimi-Dev-72B` is the exception and
does ship `LICENSE.md`.

## The instrument, which does exist

The grant is real and was retrieved and read, not inferred from the tag:

| | |
|---|---|
| Location | `github.com/MoonshotAI/Kimi-Linear`, **master** branch |
| Pinned at | commit `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3` |
| GitHub API `license.spdx_id` | `MIT` |
| Copyright line | `Copyright (c) 2025 Moonshot AI` |
| Text | the standard unmodified MIT License |

The text carries no weights carve-out, no field-of-use restriction, and no
non-commercial clause. It explicitly grants the rights to use, copy, modify,
merge, publish, distribute, sublicense and **sell** copies, subject to including
the copyright notice and permission notice.

Note the `main` branch 404s for this file. The default branch is `master`, and
an early check that assumed `main` wrongly concluded no instrument existed
anywhere.

## The decision

Our packaging preflight refuses to open a mission on a model whose declared
licence has no instrument file **in the repo the licence applies to**. That rule
exists because a tag is metadata anyone can type, and it has previously been
observed pointing at nothing. It is a good rule.

Here it produces a false negative: the instrument exists, in the publisher's own
companion repository, and says what the tag says.

Presented with that evidence and the alternative of waiting for the publisher to
mirror the file, the founder chose, on 2026-07-29, to **accept the
companion-repository instrument as sufficient**.

That is a founder decision, not an engineering one, and it is recorded here
verbatim rather than being folded silently into a config value. A reviewer who
disagrees can see exactly what was decided, by whom, on what evidence.

## What this obliges us to do

MIT permits redistribution provided the copyright notice and the permission
notice travel with the software. Any package or artifact we distribute that
contains or derives from Moonshot's code or weights therefore carries the MIT
notice and the `Copyright (c) 2025 Moonshot AI` line, attributed to Moonshot and
not to us. See `reference/PROVENANCE.md` for the file-by-file position on
vendored material, which is a separate question with separate terms: the
vendored K3 reference files split Apache-2.0 for LLaVA and DeepSeek derived code
from the Kimi K3 License for the remainder.

## The cleaner resolution, still worth pursuing

If Moonshot adds the same `LICENSE` file to the HuggingFace model repos, the
ambiguity disappears entirely and preflight passes mechanically with no judgment
call in the record. That request costs one discussion post per repo and remains
worth making regardless of the decision above.
