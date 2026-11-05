# Parser allocation extension

Pinned source: unsafe-libyaml 0.2.11, registry checksum
673aac59facbab8a9007c7f6108d11f63b603f7cabff99fabf650fea5c32b861.

`admitted::ParserAllocator` is an opt-in, thread-bound parser session. Its
initialize/parse methods install the captured core admission context and report
allocation refusal through `YAML_MEMORY_ERROR`. The intrusive allocation header
retains each allocation's reservation without a separately growing registry.
Replacement buffers are admitted while the old buffer remains live. Null checks
stop parser/scanner string, stack, queue, and raw pointer writes. Four formerly
void scanner helpers now propagate `Success` failure.

Parser initialization zeroes the entire parser first. After any failure callers
must destroy initialized events and the parser before dropping the session.
The session then reclaims local scratch abandoned by failure propagation. No
raw event payload may outlive the session. Session methods are unsafe because
raw parser/event lifetimes remain the caller's responsibility. The HTTP wrapper
copies successful event payloads into its separately admitted owned text.

This extension does not expose the fallible session to emitter, document-loader,
or general event-initialization APIs. Those default entry points retain the
upstream abort-on-physical-allocation-failure behavior. Existing public parser
entry points outside the session retain their historical behavior. No panic
unwinding, release-profile change, null-buffer continuation, YAML exclusion, or
fallback parser is used.

Source parser checks only; no local builds or tests were run.
