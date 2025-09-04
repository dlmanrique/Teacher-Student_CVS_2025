# This fiel is a test if swim-X model can make infernece in CPU
import torch
import torch.nn as nn
import timm
import time

# Modelos SwinV2 a probar
model_names = [
'swinv2_cr_large_384'
]


device = "cpu"
batch_size = 16
num_images = 200

# Crear imágenes random (ruido gaussiano) con tamaño 3x256x256


results = {}

for name in model_names:
    print(f"\n🔎 Probando modelo: {name}")
    if '384' in name:
        images = torch.randn(num_images, 3, 384, 384).to(device)
    elif '224' in name:
        images = torch.randn(num_images, 3, 224, 224).to(device)
    else:
        images = torch.randn(num_images, 3, 256, 256).to(device)

    model = timm.create_model(name, pretrained=True).to(device)
    model.eval()

    # Warm-up (importante para tiempos más realistas en GPU)
    with torch.no_grad():
        _ = model(images[:batch_size])

    # Medir tiempo
    start = time.time()
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch = images[i:i+batch_size]
            _ = model(batch)
    end = time.time()

    total_time = end - start
    avg_time = total_time / num_images
    results[name] = {"total_time": total_time, "avg_time_per_image": avg_time}
    print(results)

# 📊 Resultados
print("\n📊 Benchmark inferencia SwinV2:")
for name, metrics in results.items():
    print(f"{name}: {metrics['total_time']:.2f}s total, {metrics['avg_time_per_image']*1000:.2f} ms/imagen")

