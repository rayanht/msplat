"""Train 3DGS on mipnerf360 garden — matches msplat CLI output."""

import msplat

dataset = msplat.load_dataset(
    "datasets/mipnerf360/garden",
    downscale_factor=1.0,
    eval_mode=True,
    test_every=8,
)
print(f"Loaded {dataset.num_train} train, {dataset.num_test} test cameras")

config = msplat.TrainingConfig(
    iterations=7000,
    num_downscales=0,
    downscale_factor=1.0,
)

trainer = msplat.GaussianTrainer(dataset, config)

def on_step(stats):
    print(
        f"step={stats.iteration:>6}  "
        f"splats={stats.splat_count:>8,}  "
        f"ms={stats.ms_per_step:.1f}"
    )

trainer.train(on_step, callback_every=100)

trainer.export_ply("garden.ply")
print(f"\nSaved garden.ply ({trainer.splat_count:,} gaussians)")

metrics = trainer.evaluate()
print(f"\n=== Evaluation ({metrics['num_test']} test views) ===")
print(f"  PSNR:  {metrics['psnr']:.4f}")
print(f"  SSIM:  {metrics['ssim']:.4f}")
print(f"  L1:    {metrics['l1']:.4f}")
print(f"  Gaussians: {metrics['num_gaussians']:,}")

msplat.cleanup()
