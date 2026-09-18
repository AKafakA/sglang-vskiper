# Reproducing vSkipper

Choose the route that matches your task:

- **Reproduce the reported numerical outputs:** use the reviewer pack and
  [stage 3](03-run-analysis.md). The included source and evidence run on CPU.
- **Run GPU measurements:** prepare the environment and checkpoint assets in
  [stage 1](01-environment.md), then follow [stage 2](02-run-experiments.md).

The pack holds reproduction materials: evidence, analysis code, reference
outputs and instructions. Model weights and GPU environments are supplied
separately. Each pack release identifies its source commit and hashes its
bundled inputs. Use the matching reference revision when reproducing through
the repository.

[Integrity gates](gates.md), [integration map](integration.md), and
[file relocation and cleanup](removed.md) explain how the implementation and
measurement artifacts fit together.
