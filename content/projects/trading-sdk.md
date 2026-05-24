## The problem

We had a Python SDK, but adoption was lower than expected and the README hadn't been touched in a year.

## What I did

Rewrote documentation around the user's first-five-minutes experience. Added a daily integration test that hits the real API and posts to Slack on regression. Folded examples into the test suite so they can't drift.

## What it taught me

An SDK's value is mostly its README. Code is secondary.
