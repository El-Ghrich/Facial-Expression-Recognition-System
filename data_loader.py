import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = (48, 48)
COLOR_MODE = 'grayscale'
CLASS_MODE = 'categorical'
VAL_SPLIT  = 0.2


def create_generators(path='dataset', batch_size=64):
    train_path = os.path.join(path, 'train')
    test_path  = os.path.join(path, 'test')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=VAL_SPLIT
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_path, target_size=IMG_SIZE, color_mode=COLOR_MODE,
        batch_size=batch_size, class_mode=CLASS_MODE,
        subset='training', shuffle=True, seed=42
    )

    val_gen = train_datagen.flow_from_directory(
        train_path, target_size=IMG_SIZE, color_mode=COLOR_MODE,
        batch_size=batch_size, class_mode=CLASS_MODE,
        subset='validation', shuffle=False, seed=42
    )

    test_gen = test_datagen.flow_from_directory(
        test_path, target_size=IMG_SIZE, color_mode=COLOR_MODE,
        batch_size=batch_size, class_mode=CLASS_MODE,
        shuffle=False
    )

    return train_gen, val_gen, test_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FER data loading & augmentation pipeline')
    parser.add_argument('--path', type=str, default='dataset',
                        help='Root path to dataset folder (default: dataset/)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    args = parser.parse_args()

    train_generator, val_generator, test_generator = create_generators(
        path=args.path, batch_size=args.batch_size
    )

    print("=" * 50)
    print("Class indices:", train_generator.class_indices)
    print("Classes:", list(train_generator.class_indices.keys()))
    print(f"Train batches: {len(train_generator)}")
    print(f"Val batches:   {len(val_generator)}")
    print(f"Test batches:  {len(test_generator)}")

    for name, gen in [("Train", train_generator),
                       ("Val",   val_generator),
                       ("Test",  test_generator)]:
        x, y = next(iter(gen))
        print(f"{name} -> X: {x.shape}, y: {y.shape}")
    print("=" * 50)
