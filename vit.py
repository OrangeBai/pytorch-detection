from vit_pytorch import ViT

model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

for name, module in model.named_modules():
    print(name)
    print(module)

print(1)

