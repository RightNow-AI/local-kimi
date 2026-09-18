# Third-party reference provenance

This directory contains third-party Moonshot material retained for test-oracle
and architecture-reference purposes. These files are not imported by `k3/` at
runtime and are not included in the `k3` wheel.

The project's Apache-2.0 licence does not relicense these files. Their upstream
terms continue to apply.

## Vendored files

### Kimi K3

Upstream repository: `moonshotai/Kimi-K3`

Audited revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`

| Local file | Upstream relationship | Header declaration | Purpose here |
| --- | --- | --- | --- |
| `encoding_k3.py` | Exact upstream text after CRLF-to-LF normalization | No licence declaration in the file header | Golden XTML encoder used by oracle tests |
| `tokenization_kimi.py` | Exact upstream text after CRLF-to-LF normalization | No licence declaration in the file header | Moonshot tokenizer behavior used by oracle tests |
| `modeling_kimi_k3.py` | Upstream text with one import line reformatted | Apache-2.0 for LLaVA-derived code; Kimi K3 License for other code | Architecture reference |
| `modeling_kimi_linear.py` | Upstream text with an unused `auto_docstring` import removed and import spacing changed | Apache-2.0 for DeepSeek-derived code; Kimi K3 License for other code | Architecture reference |
| `configuration_kimi_k3.py` | Exact upstream text after CRLF-to-LF normalization | No licence declaration in the file header | Architecture configuration reference |
| `config.json` | Exact upstream text after CRLF-to-LF normalization | No file-level declaration | Pinned Kimi K3 configuration reference |

Source URLs at the audited revision:

- `https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/encoding_k3.py`
- `https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/tokenization_kimi.py`
- `https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_k3.py`
- `https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py`
- `https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/configuration_kimi_k3.py`
- `https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json`

Licence URL fetched with Python `urllib` on 2026-07-28:

`https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/LICENSE`

The fetched licence is the Kimi K3 License, copyright 2026 Moonshot AI. It
expressly grants the rights to use, copy, modify, merge, publish, distribute,
sublicense, sell, and create derivative works. Redistribution in a public
repository is therefore permitted, provided its copyright and permission
notice is retained and the additional conditions in sections 2 through 4 are
followed.

### Kimi K2 Thinking chat template

Local file: `kimi_k2_thinking_chat_template.jinja`

Upstream repository: `moonshotai/Kimi-K2-Thinking`

Audited revision: `a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55`

The local file is exact after CRLF-to-LF normalization to upstream
`chat_template.jinja`:

`https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/chat_template.jinja`

Licence URL fetched with Python `urllib` on 2026-07-28:

`https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/LICENSE`

The fetched licence is a Modified MIT License, copyright 2025 Moonshot AI. It
expressly permits copying, modification, publication, distribution,
sublicensing, and sale, provided its copyright and permission notice is
retained. It adds an attribution condition for commercial products or services
above the stated monthly active-user or monthly-revenue thresholds.

## Kimi-Linear licence check

The requested audit also checked
`moonshotai/Kimi-Linear-48B-A3B-Instruct` at revision
`e1df551a447157d4658b573f9a695d57658590e9`.

- `LICENSE` returned HTTP 404.
- `LICENSE.md` returned HTTP 404.
- The model card declares `license: mit` in its metadata.
- The linked `MoonshotAI/Kimi-Linear` GitHub repository uses `master` as its
  default branch and provides a standard MIT License there, copyright 2025
  Moonshot AI.

Model card URL:

`https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/README.md`

Linked code-repository licence URL, checked at commit
`8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`:

`https://github.com/MoonshotAI/Kimi-Linear/blob/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3/LICENSE`

The model card's MIT declaration and the linked code repository's licence
agree. The Hugging Face model repository itself does not include the licence
text at `LICENSE` or `LICENSE.md`. None of the vendored files listed above is
sourced from that Kimi-Linear model repository. The Kimi K3 repository is the
source of the similarly named `modeling_kimi_linear.py` in this directory.

## Kimi K3 License text retrieved

The text below reproduces the fetched notice with ASCII punctuation.

```text
Kimi K3 License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person (the "Licensee")
obtaining a copy of this software, including the model weights, parameters,
configuration files, inference and training code, and associated documentation
(collectively, the "Software"), to deal in the Software without restriction.
This includes, without limitation, the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software; to run,
deploy, fine-tune, or otherwise modify the Software and create derivative works
from it; and to permit persons to whom the Software is furnished to do so, in
each case subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. Licensee's use of the
Software must comply with applicable laws and regulations.

2. "Model as a Service" means giving a third party access to language model
inference or fine-tuning (e.g., via API) in a manner that allows such third
party to exercise meaningful control over the inputs, parameters, or training
data. This does not include (a) end-user products with model capabilities solely
embedded within specific features or harnesses, or (b) mere relaying of requests
to models hosted by others.

If the Licensee or any of its affiliates operates a Model as a Service business,
and the aggregate revenue of the Licensee and its affiliates exceeds 20 million
US dollars (or the equivalent in other currencies) in total over any consecutive
12 months, the Licensee must enter into a separate agreement with Moonshot AI
before using the Software or its derivative works for any commercial purpose.

3. If the Software (or any derivative works thereof) is used for any of the
Licensee's commercial products or services that have more than 100 million
monthly active users, or more than 20 million US dollars (or equivalent in other
currencies) in monthly revenue, "Kimi K3" must be prominently displayed on the
user interface of such product or service.

4. The requirements set forth in Sections 2 and 3 do not apply to: (a) internal
use of the Software, defined as any use that does not make the Software, its
outputs, or its underlying capabilities available to third parties; or (b) any
use of the Software accessed through Moonshot AI's official products or
certified inference partners.

5. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS"
BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL MOONSHOT AI OR ITS AFFILIATES OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For any questions regarding this license, please contact <license@moonshot.ai>.
```

## Modified MIT License text retrieved

The text below reproduces the fetched notice with ASCII punctuation.

```text
Modified MIT License

Copyright (c) 2025 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2" on the user interface of such product or service.
```

## Kimi-Linear MIT License text retrieved

```text
MIT License

Copyright (c) 2025 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
