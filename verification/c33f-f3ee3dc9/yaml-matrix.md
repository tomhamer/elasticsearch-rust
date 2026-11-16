# ES9 YAML parity with module-specific topology: pr3-consolidated80-yaml

| Suite | Pass | Fail | Skip |
| --- | ---: | ---: | ---: |
| oss | 2657 | 0 | 53 |
| main | 1729 | 5 | 104 |
| sec | 195 | 0 | 0 |
| plug | 1692 | 0 | 35 |

## Failures by directory

- main `rollup`: 5

## Failures by cause

- 5 × rollup.start_job POST /_rollup/job/foo/_start -> N illegal_argument_exception: Cannot start task for Rollup Jo
