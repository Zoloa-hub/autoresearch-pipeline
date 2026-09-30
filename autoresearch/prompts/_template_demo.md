# Template-engine contract demo (used by `tests/test_prompts.py`).

This file is *not* a model-facing prompt. It is a fixture that exercises every
piece of template syntax `PromptLibrary` supports, so the engine's behaviour is
pinned by a committed file rather than an inline string.

## 1. Variable substitution (whitespace-tolerant)

- `{{ direction }}`
- `{{direction}}`
- `{{   direction   }}`
- Unknown -> empty and warns: `[{{ definitely_not_a_variable }}]`
- Nested path: `{{ plan.baseline.name }}`

## 2. LaTeX and JSON braces must survive verbatim

```
\begin{tabular}{ll}
\frac{a}{b}
\cite{key}
\section{a{b}c}
{"queries": ["x"], "rationale": "y"}
```

## 3. `{% if %}`

{% if flag %}
FLAG:TRUE
{% else %}
FLAG:FALSE
{% endif %}

{% if empty_list %}
LIST:TRUE
{% else %}
LIST:FALSE
{% endif %}

{% if direction %}
HAVE_DIRECTION
{% endif %}

Nested condition:

{% if outer %}
{% if inner %}OUTER-INNER-BOTH{% else %}OUTER-ONLY{% endif %}
{% endif %}

## 4. `{% for %}`

Scalars:

{% for item in scalars %}
- item={{ item }} index={{ loop.index }} index1={{ loop.index1 }} first={{ loop.first }}
{% endfor %}

Mappings (dotted access):

{% for p in papers %}
- {{ p.id }} :: {{ p.title }}
{% endfor %}

## 5. `{{ language }}` is injected automatically

language={{ language }}

END-OF-DEMO
