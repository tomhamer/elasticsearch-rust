// Test fixture generator. Calls the pinned upstream implementation, never Rustic.
import java.util.Arrays;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;

public class EncodedScoreOracle {
  static final class Values extends QuantizedByteVectorValues {
    final float[] centroid; final byte[] packed; final ScalarEncoding encoding;
    final OptimizedScalarQuantizer quantizer;
    final OptimizedScalarQuantizer.QuantizationResult correction;
    Values(float[] c, byte[] codes, ScalarEncoding e, OptimizedScalarQuantizer q,
           OptimizedScalarQuantizer.QuantizationResult correction) {
      centroid=c; encoding=e; quantizer=q; this.correction=correction;
      packed=new byte[e.getDocPackedLength(c.length)];
      if(e==ScalarEncoding.SEVEN_BIT) System.arraycopy(codes,0,packed,0,codes.length);
      else if(e==ScalarEncoding.PACKED_NIBBLE) {
        for(int i=0;i<packed.length;i++) packed[i]=(byte)((codes[i]<<4)|codes[i+packed.length]);
      } else OptimizedScalarQuantizer.packAsBinary(codes,packed);
    }
    public int dimension(){return centroid.length;}
    public int size(){return 1;}
    public byte[] vectorValue(int ord){if(ord!=0)throw new IllegalArgumentException();return packed;}
    public Values copy(){return this;}
    public OptimizedScalarQuantizer getQuantizer(){return quantizer;}
    public ScalarEncoding getScalarEncoding(){return encoding;}
    public float[] getCentroid(){return centroid;}
    public float getCentroidDP(){return VectorUtil.dotProduct(centroid,centroid);}
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int ord){return correction;}
  }
  static String correction(OptimizedScalarQuantizer.QuantizationResult c){
    return "["+c.lowerInterval()+","+c.upperInterval()+","+c.additionalCorrection()+","+c.quantizedComponentSum()+"]";
  }
  static float[] row(int dims,int seed){
    float[] v=new float[dims];
    for(int i=0;i<dims;i++) v[i]=(((i*37+seed*19)%127)-63)/64f;
    return v;
  }
  public static void main(String[] args)throws Exception {
    if(!Version.LATEST.toString().equals("10.5.1"))throw new IllegalStateException("Requires Lucene 10.5.1, got "+Version.LATEST);
    System.out.println("{\"lucene\":\"10.5.1\",\"oracle\":\"Lucene104ScalarQuantizedVectorScorer\",\"cases\":[");
    boolean first=true;
    for(int bits:new int[]{7,4,1})for(String name:new String[]{"l2","dot","cosine","mip"})for(int seed=0;seed<16;seed++){
      int dims=seed%2==0?64:96;
      var sim=name.equals("l2")?VectorSimilarityFunction.EUCLIDEAN:name.equals("mip")?VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT:VectorSimilarityFunction.DOT_PRODUCT;
      // ES normalizes cosine at mapping time, then scores it as DOT_PRODUCT.
      float[] doc=row(dims,seed), query=row(dims,seed+7), centroid=row(dims,seed+3);
      if(name.equals("dot")||name.equals("cosine")){VectorUtil.l2normalize(doc);VectorUtil.l2normalize(query);}
      for(int i=0;i<dims;i++)centroid[i]*=0.03125f;
      var encoding=ScalarEncodingFor(bits);
      var quantizer=new OptimizedScalarQuantizer(sim);
      byte[] dc=new byte[dims],qc=new byte[dims];
      var d=quantizer.scalarQuantize(doc.clone(),dc,(byte)bits,centroid);
      var q=quantizer.scalarQuantize(query.clone(),qc,encoding.getQueryBits(),centroid);
      var values=new Values(centroid,dc,encoding,quantizer,d);
      float score=new Lucene104ScalarQuantizedVectorScorer(null).getRandomVectorScorer(sim,values,query).score(0);
      if(!Float.isFinite(score))throw new IllegalStateException("nonfinite oracle score");
      if(!first)System.out.println(",");first=false;
      System.out.print("{\"bits\":"+bits+",\"sim\":\""+name+"\",\"centroid\":"+Arrays.toString(centroid)+",\"centroid_dp\":"+values.getCentroidDP()+",\"query_codes\":"+Arrays.toString(qc)+",\"doc_codes\":"+Arrays.toString(dc)+",\"query_correction\":"+correction(q)+",\"doc_correction\":"+correction(d)+",\"score\":"+score+"}");
    }
    System.out.println("]}");
  }
  static QuantizedByteVectorValues.ScalarEncoding ScalarEncodingFor(int bits){return bits==7?QuantizedByteVectorValues.ScalarEncoding.SEVEN_BIT:bits==4?QuantizedByteVectorValues.ScalarEncoding.PACKED_NIBBLE:QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;}
}
