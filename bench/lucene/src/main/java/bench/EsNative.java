package bench;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/** Optional ES distribution classes, without building or starting an ES node. */
public final class EsNative {
    private static boolean initialized;
    static synchronized void initialize() throws Exception {
        if (!initialized) {
            String home = System.getenv("ES_NATIVE_HOME");
            if (home == null) throw new IllegalArgumentException("ES_NATIVE_HOME must point to the matching ES distribution");
            System.setProperty("es.path.home", home);
            Class.forName("org.elasticsearch.common.logging.LogConfigurator").getMethod("configureESLogging").invoke(null);
            Class<?> settings = Class.forName("org.elasticsearch.common.settings.Settings");
            Class.forName("org.elasticsearch.common.logging.LogConfigurator").getMethod("configureWithoutConfig", settings)
                .invoke(null, settings.getField("EMPTY").get(null));
            Object provider = Class.forName("org.elasticsearch.simdvec.ESVectorizationProvider").getMethod("getInstance").invoke(null);
            System.err.println("ES native vector provider: " + provider.getClass().getName());
            initialized = true;
        }
    }
    public static KnnVectorsFormat format(int m, int construction) throws Exception {
        initialize();
        Class<?> element = Class.forName("org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper$ElementType");
        Object floats = java.util.Arrays.stream(element.getEnumConstants())
            .filter(value -> ((Enum<?>) value).name().equals("FLOAT")).findFirst().orElseThrow();
        return (KnnVectorsFormat) Class.forName("org.elasticsearch.index.codec.vectors.es93.ES93HnswVectorsFormat")
            .getConstructor(int.class, int.class, element).newInstance(m, construction, floats);
    }

    public static Query query(String field, float[] vector, int k, int candidates, Query filter) throws Exception {
        return query(field, vector, k, candidates, filter, true);
    }

    public static Query query(String field, float[] vector, int k, int candidates, Query filter, boolean earlyTermination) throws Exception {
        initialize();
        return (Query) Class.forName("org.elasticsearch.search.vectors.ESKnnFloatVectorQuery")
            .getConstructor(String.class, float[].class, int.class, int.class, Query.class, KnnSearchStrategy.class, boolean.class)
            .newInstance(field, vector, k, candidates, filter, new KnnSearchStrategy.Hnsw(60), earlyTermination);
    }
}
