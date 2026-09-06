# Third-party sources and data

This software archive redistributes none of the benchmark records and none of the `label-delay-exp` source tree. The launcher fetches the following pinned material into `data/` at run time.

| Material | Frozen identifier | Use in this package | Terms/source |
|---|---|---|---|
| Label Delay in Online Continual Learning code | `5cd6f59e48e8015ecd540f56c08449cb05846214` | Yearbook loader, official split interpretation, chronological within-year stream, delay convention | [botcs/label-delay-exp](https://github.com/botcs/label-delay-exp); inspect the fetched tree and its file headers |
| Yearbook dating split repository | `752c48aa420fd47cd25a5043d8305f3d636ff86c` | F/M train/test lists when a list is absent from the archive | [katerakelly/yearbook-dating](https://github.com/katerakelly/yearbook-dating); UC Berkeley educational/research/non-profit notice |
| Yearbook images | Runtime download; actual SHA-256 recorded | Chronological image benchmark | URL published in the NeurIPS authors' README; images are not included here |
| Criteo attribution dataset | revision `904188a63cbad78bee43cd26ff5ee4ac77903986`; object SHA-256 `94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98` | Click-filtered conversion stream and feedback timing | [dataset card](https://huggingface.co/datasets/criteo/criteo-attribution-dataset); CC BY-NC-SA 4.0 |
| ResNet-18 ImageNet-1K V1 weights | torchvision `ResNet18_Weights.IMAGENET1K_V1` | Frozen Yearbook token producer | [torchvision model documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) |

Users are responsible for confirming that their use satisfies the benchmark licenses and terms, particularly the non-commercial restrictions on Criteo and the Yearbook materials.

