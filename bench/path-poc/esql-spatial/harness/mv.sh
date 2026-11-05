p=$1
curl -s -XDELETE localhost:$p/mvt >/dev/null
curl -s -XPUT localhost:$p/mvt -H content-type:application/json -d '{"mappings":{"properties":{"loc":{"type":"geo_point"},"n":{"type":"long"}}}}' >/dev/null
curl -s -XPOST "localhost:$p/mvt/_bulk?refresh=true" -H content-type:application/x-ndjson --data-binary $'{"index":{}}\n{"n":1,"loc":[{"lat":50,"lon":5},{"lat":10,"lon":40}]}\n{"index":{}}\n{"n":2,"loc":{"lat":51,"lon":6}}\n{"index":{}}\n{"n":3,"loc":[{"lat":10,"lon":40},{"lat":11,"lon":41}]}\n' >/dev/null
P='TO_GEOSHAPE(\"POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))\")'
for q in "FROM mvt | STATS c = ST_CENTROID_AGG(loc)" "FROM mvt | WHERE ST_INTERSECTS(loc, $P) | KEEP n | SORT n" "FROM mvt | EVAL x = ST_INTERSECTS(loc, $P) | KEEP n, x | SORT n" "FROM mvt | EVAL x = ST_WITHIN(loc, $P), y = ST_DISJOINT(loc, $P) | KEEP n, x, y | SORT n" "FROM mvt | EVAL d = ST_DISTANCE(loc, TO_GEOPOINT(\"POINT(7 55)\")) | KEEP n, d | SORT n" "FROM mvt | WHERE ST_WITHIN(loc, $P) | KEEP n | SORT n" "FROM mvt | WHERE ST_DISJOINT(loc, $P) | KEEP n | SORT n"; do
  echo "== $q"; curl -s -H content-type:application/json "localhost:$p/_query?format=txt" -d "{\"query\":\"$q\"}"; echo
done
