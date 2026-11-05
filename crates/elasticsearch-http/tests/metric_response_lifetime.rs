//! The real HTTP handler owns scripted-metric output through serialization and
//! body delivery. InProcessJson must not let a charged naked Value escape.
use axum::{body::{Body,to_bytes},http::{Method,Request}};
use elasticsearch_engine::node::Node;
use elasticsearch_http::{InProcessJson,JsonResponse};
use serde_json::{json,Value};
use tower::ServiceExt;

#[tokio::test]
async fn metric_output_charge_survives_handler_and_releases_on_body_completion_or_drop() {
    let dir=tempfile::tempdir().unwrap();
    let node=Node::open(dir.path()).unwrap();
    let router=elasticsearch_http::router(node.clone());
    for (method,path,value) in [
        (Method::PUT,"/metric",json!({"mappings":{"properties":{"n":{"type":"long"}}}})),
        (Method::PUT,"/metric/_doc/1?refresh=true",json!({"n":1})),
    ] {
        let response=router.clone().oneshot(Request::builder().method(method).uri(path).header("content-type","application/json").body(Body::from(value.to_string())).unwrap()).await.unwrap();
        assert!(response.status().is_success());
        let _=to_bytes(response.into_body(),usize::MAX).await.unwrap();
    }
    let before=node.breakers.request.reserved();
    for in_process in [false,true] {
        for discard in [false,true] {
            let value=json!({"size":0,"aggs":{"m":{"scripted_metric":{
                "init_script":"state.n=0", "map_script":"state.n+=1",
                "combine_script":"return state.n", "reduce_script":"return [1,2,3]"
            }}}});
            let mut request=Request::builder().method(Method::POST).uri("/metric/_search").header("content-type","application/json").body(Body::from(value.to_string())).unwrap();
            if in_process {request.extensions_mut().insert(InProcessJson);}
            let response=router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status().as_u16(),200);
            assert!(node.breakers.request.reserved()>before,"successful handler must retain metric output charge");
            assert!(response.extensions().get::<JsonResponse>().is_none(),"charged Value cannot escape via an unowned extension");
            if discard {drop(response);} else {
                let bytes=to_bytes(response.into_body(),usize::MAX).await.unwrap();
                let result:Value=serde_json::from_slice(&bytes).unwrap();
                assert_eq!(result["aggregations"]["m"]["value"],json!([1,2,3]));
            }
            assert_eq!(node.breakers.request.reserved(),before,"completion and cancellation both release output charge");
        }
    }
}
