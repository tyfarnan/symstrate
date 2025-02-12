Overview

Can techniques from automatic theorem proving be leveraged for program synthesis? Can algorithm discovery techniques be generalized for program synthesis? By utilizing arithmetization techniques, programs can be encoded as a set of polynomial constraints that represent the gates of an arithmetic circuit? In this form, programs could be synthesized programs via a bottom up approach where simple gates are learned first, and then more complex gates are learned by composing simpler gates.

Symstrate aims to enable the exploration of symbolic regression for program synthesis via generative benchmarks that allow us to formally evaluate a model's ability to mine & compose symbolic abstractions. Can formal verification techniques be borrowed to build a synthetic data generator that generates input-output examples with semantically equivalent "function" AND syntactic program diversity?

This work aims to provide the following contributions:

1. A forward pipeline that generates arithmetic circuits and polynomial constrain systems from a restricted python DSL. We can utilize python's AST to generate these circuits via DAG traversal.
2. A backward pipeline that generates programs from a set of polynomial constraints, allowing for algabreic rewirings preserve semantic equivalence while producing syntax diversity, encouraging models to learn and reuse the actual undelrying circuit structures.

Roadmap:

1. Build a program translater that takes leetcode problems and rewrites them using the restricted pythoh DSL that can be statically analyzed with the abstract syntax tree (AST).
2. Build a tool that takes the AST and generates an arithmetic circuit and polynomial constraint system.
3. Generate a small number of toy examples (leetcode problems statement and input ouput examples). Evaluate whether the model can generate the polynomials constraint system.

Questions:
1. How does O3 do on existing symbolic regression benchmarks?
2. How well can O3 generate the polynomials constraint system?
3. How well can O3 generate the programs from the polynomials constraint system?
4. How well can O3 learn the underlying arithmetic circuit given the polynomials constraint system?
4. How well can O3 learn the underlying arithmetic circuit given the program?
5. How well can O3 learn the underlying arithmetic circuit given the polynomials constraint system?



References:

- [O3 ARC-AGI breakthrough](https://arcprize.org/blog/oai-o3-pub-breakthrough)
- [alphageometry](https://github.com/google-deepmind/alphageometry)
- [alphatensor](https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/)
- [PLONK arithmetization](https://www.youtube.com/watch?v=0M0pAubEjz8)
- [DreamCoder](https://arxiv.org/abs/2006.08381)
- [SR tutorial](https://nmakke.github.io/SRTutorial_kdd24/)
- [equation tree generator](https://autoresearch.github.io/equation-tree/)


Ideas:

Borrowing techniques from zero-knowledge proofs, there is potential to verify the correctness of programs without actually executing them.


