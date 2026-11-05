# Finalized outgoing header ownership

Based on published `http` 1.5.0. Source and licenses retained.

Adds an opt-in fallible finalized-map constructor. The library computes its
concrete Pos/Bucket/ExtraValue storage before allocation; initialized randomized
hashing prevents collision-triggered expansion during the bounded insertion.
Payload names accept validated shared lowercase Bytes, preserving text owners.
Backing storage ownership follows both HeaderMap and its consuming IntoIter.

This is an audited engine outgoing-path boundary, not a replacement for the
library's generic allocation contract. Existing default network/dependency maps
remain unchanged. An admitted map must not use infallible deep Clone or growth;
engine changes must reconstruct it through the fallible constructor. Individual
header values/names can be extracted or cloned because their Bytes retain owners.
No infallible API was changed to panic on breaker refusal.

The audited server configuration is Axum's default HTTP/1 serving path. Hyper
1.11.1 HTTP/1 drains the map and emits synthesized Date/content-length fields
directly to its wire buffer. HTTP/2 is not enabled by the workspace server
features; enabling it requires a new audit because its server path inserts Date
and content-length into this map. Do not extend this ownership claim to HTTP/2
or middleware that deep-clones or grows a finalized map.
