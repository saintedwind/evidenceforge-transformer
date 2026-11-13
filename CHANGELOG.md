# Changelog

## 0.3.1

Short citations now select a 280-character window using weighted coverage of
complete query terms. This packages the citation improvement from commit
`8a80d642cef47f779e2bbb6f45b2a24f111d2929`; document ranking, access filtering,
and the separate 800-character generation context are unchanged.

In the recorded development evaluation on 1,000 PersianQuAD questions over 35
contexts, complete annotated answers appeared in the top-five citations for
736 questions, versus 410 with the 0.3.0 wheel. There were 382 gains and 56
losses, and local p95 latency increased. These results measure citation spans,
not generated-answer accuracy or generalization. See
[the evaluation and reproduction instructions](reports/CITATION_EVALUATION.md).

## 0.3.0

Published baseline with persistent corpus storage, access-filtered retrieval,
and bounded local source selection.
