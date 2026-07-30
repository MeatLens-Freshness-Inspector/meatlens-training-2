from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_training_augmentation(preset: str = "geometry_only_v1") -> tf.keras.Sequential:
    base_layers: list[tf.keras.layers.Layer] = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08)),
        layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
    ]

    if preset == "geometry_only_v1":
        return tf.keras.Sequential(base_layers, name="training_augmentation")

    if preset == "conservative_v1":
        return tf.keras.Sequential(
            [
                *base_layers,
                layers.Lambda(
                    lambda tensor: tf.clip_by_value(tf.image.random_brightness(tensor, max_delta=12.0), 0.0, 255.0),
                    name="random_brightness",
                ),
                layers.Lambda(
                    lambda tensor: tf.clip_by_value(tf.image.random_contrast(tensor, lower=0.9, upper=1.1), 0.0, 255.0),
                    name="random_contrast",
                ),
            ],
            name="training_augmentation",
        )

    raise ValueError(f"Unsupported augmentation preset: {preset}")
