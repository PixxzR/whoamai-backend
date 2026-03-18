"""Architectures TensorFlow/Keras (MobileNetV2) pour modèles légers mobile/TFLite."""

import tensorflow as tf
from tensorflow import keras


# =============================================================================
# SPECIALIZED : 3 modèles MobileNetV2 séparés
# =============================================================================

def create_age_model(input_shape=(224, 224, 3)) -> keras.Model:
    """MobileNetV2 pour la régression d'âge."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(1, activation="linear", name="age")(x)
    return keras.Model(inputs=base.input, outputs=output, name="age_mobilenetv2")


def create_gender_model(input_shape=(224, 224, 3), num_classes: int = 2) -> keras.Model:
    """MobileNetV2 pour la classification de genre."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(num_classes, activation="softmax", name="gender")(x)
    return keras.Model(inputs=base.input, outputs=output, name="gender_mobilenetv2")


def create_ethnicity_model(input_shape=(224, 224, 3), num_classes: int = 5) -> keras.Model:
    """MobileNetV2 pour la classification d'ethnicité."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(num_classes, activation="softmax", name="ethnicity")(x)
    return keras.Model(inputs=base.input, outputs=output, name="ethnicity_mobilenetv2")


# =============================================================================
# MULTITASK : 1 MobileNetV2 partagé, 3 têtes de sortie
# =============================================================================

def create_multitask_model(
    input_shape=(224, 224, 3),
    num_genders: int = 2,
    num_ethnicities: int = 5,
) -> keras.Model:
    """MobileNetV2 avec 3 têtes : âge, genre, ethnicité."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    shared = keras.layers.GlobalAveragePooling2D(name="shared_gap")(base.output)

    # Tête âge
    age_x = keras.layers.Dense(128, activation="relu", name="age_dense")(shared)
    age_output = keras.layers.Dense(1, activation="linear", name="age")(age_x)

    # Tête genre
    gender_x = keras.layers.Dense(64, activation="relu", name="gender_dense")(shared)
    gender_output = keras.layers.Dense(num_genders, activation="softmax", name="gender")(gender_x)

    # Tête ethnicité
    eth_x = keras.layers.Dense(128, activation="relu", name="ethnicity_dense")(shared)
    eth_output = keras.layers.Dense(num_ethnicities, activation="softmax", name="ethnicity")(eth_x)

    return keras.Model(
        inputs=base.input,
        outputs={"age": age_output, "gender": gender_output, "ethnicity": eth_output},
        name="multitask_mobilenetv2",
    )


# =============================================================================
# TRANSFER : MobileNetV2 avec couches early gelées
# =============================================================================

def create_transfer_model(
    input_shape=(224, 224, 3),
    num_genders: int = 2,
    num_ethnicities: int = 5,
    freeze_layers: int = 100,
) -> keras.Model:
    """MobileNetV2 avec les premières couches gelées pour transfer learning."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Geler les premières couches
    for layer in base.layers[:freeze_layers]:
        layer.trainable = False
    for layer in base.layers[freeze_layers:]:
        layer.trainable = True

    shared = keras.layers.GlobalAveragePooling2D(name="shared_gap")(base.output)

    # 3 têtes identiques au multitask
    age_x = keras.layers.Dense(128, activation="relu", name="age_dense")(shared)
    age_output = keras.layers.Dense(1, activation="linear", name="age")(age_x)

    gender_x = keras.layers.Dense(64, activation="relu", name="gender_dense")(shared)
    gender_output = keras.layers.Dense(num_genders, activation="softmax", name="gender")(gender_x)

    eth_x = keras.layers.Dense(128, activation="relu", name="ethnicity_dense")(shared)
    eth_output = keras.layers.Dense(num_ethnicities, activation="softmax", name="ethnicity")(eth_x)

    return keras.Model(
        inputs=base.input,
        outputs={"age": age_output, "gender": gender_output, "ethnicity": eth_output},
        name="transfer_mobilenetv2",
    )
