## The problem

Analysts at DeepGrid wanted plots from English-language questions, without writing matplotlib by hand.

## What I did

Built a small planner that maps natural-language requests to a chart spec, then renders via matplotlib. Wrapped behind a FastAPI service consumed by the UI team.

## What it taught me

Most "natural language" problems are really "how strict can your schema be while still feeling natural?" problems.
