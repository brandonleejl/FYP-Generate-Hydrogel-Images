import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- Configuration & Hyperparameters ---
IMG_SIZE = 256
CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.0002
BETA_1 = 0.5

# --- Paths (Google Drive Integration) ---
BASE_PATH = '/content/drive/MyDrive/FYP_Hydrogel_Data'
IMAGES_DIR = os.path.join(BASE_PATH, 'images')
LABELS_FILE = os.path.join(BASE_PATH, 'labels.csv')
SAVED_MODELS_DIR = os.path.join(BASE_PATH, 'saved_models')
GENERATED_SAMPLES_DIR = os.path.join(BASE_PATH, 'generated_samples')

# Ensure output directories exist
for directory in [SAVED_MODELS_DIR, GENERATED_SAMPLES_DIR]:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            print(f"Warning: Could not create directory {directory}. Check Drive mounting.")

# --- Google Colab Integration ---
try:
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
except ImportError:
    print("Google Colab environment not detected. Skipping Drive mount.")

# --- Data Loading & Preprocessing ---

def load_and_preprocess_image(path, label):
    """
    Loads an image from a file path, decodes it, resizes it,
    and normalizes pixel values to the range [-1, 1].
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=CHANNELS)  # Updated to PNG
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
    return img, label

def augment_image(img, label):
    """
    Applies random horizontal and vertical flips to the image.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, label

def create_dataset():
    """
    Creates a tf.data.Dataset from the CSV file and image directory.
    """
    if not os.path.exists(LABELS_FILE):
        print(f"Error: Labels file not found at {LABELS_FILE}")
        return None

    df = pd.read_csv(LABELS_FILE)
    image_paths = [os.path.join(IMAGES_DIR, fname) for fname in df['filename']]
    labels = df['ph'].values.astype(np.float32).reshape(-1, 1)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(df)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Model Architecture ---

def build_generator():
    """
    Builds the Conditional Generator model.
    """
    noise_input = Input(shape=(LATENT_DIM,), name='noise_input')
    label_input = Input(shape=(1,), name='label_input')

    start_dim = IMG_SIZE // 32

    # Process Label
    label_embedding = layers.Dense(start_dim * start_dim * 1)(label_input)
    label_embedding = layers.Reshape((start_dim, start_dim, 1))(label_embedding)

    # Process Noise
    noise_embedding = layers.Dense(start_dim * start_dim * 256)(noise_input)
    noise_embedding = layers.Reshape((start_dim, start_dim, 256))(noise_embedding)

    x = layers.Concatenate()([noise_embedding, label_embedding])

    # Upsampling Blocks
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    output = layers.Activation('tanh')(x)

    return models.Model([noise_input, label_input], output, name="Generator")

def build_discriminator():
    """
    Builds the Conditional Discriminator model.
    """
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name='image_input')
    label_input = Input(shape=(1,), name='label_input')

    label_embedding = layers.Dense(IMG_SIZE * IMG_SIZE)(label_input)
    label_channel = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(label_embedding)

    x = layers.Concatenate()([img_input, label_channel])

    # Downsampling Blocks
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    return models.Model([img_input, label_input], output, name="Discriminator")

# --- Loss Functions & Optimizers ---

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

# --- Training Loop ---

@tf.function
def train_step(images, labels, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, test_labels, save_dir=GENERATED_SAMPLES_DIR, prefix='image_at_epoch'):
    """
    Generates a grid of images for visualization and saves to the specified directory.
    """
    # Ensure test_labels is a tensor
    if not isinstance(test_labels, tf.Tensor):
        test_labels = tf.constant(test_labels)

    predictions = model([test_input, test_labels], training=False)

    fig = plt.figure(figsize=(10, 10))
    predictions = (predictions + 1) / 2.0

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.title(f"pH: {test_labels[i, 0]:.1f}")
        plt.axis('off')

    if prefix == 'final_samples':
        filename = f"{prefix}.png"
    else:
        filename = '{}_{:04d}.png'.format(prefix, epoch)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved sample grid to {save_path}")
    return save_path

# --- Main Execution ---

if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()

    print("Models initialized.")
    generator.summary()
    discriminator.summary()

    dataset = create_dataset()

    if dataset is None:
        print("Dataset creation failed. Exiting training setup.")
    else:
        num_examples_to_generate = 16
        seed_noise = tf.random.normal([num_examples_to_generate, LATENT_DIM])
        # Generate random pH labels and convert to tf.constant
        seed_labels_np = np.random.uniform(4.0, 8.0, (num_examples_to_generate, 1)).astype(np.float32)
        seed_labels = tf.constant(seed_labels_np)

        print("Starting training...")

        for epoch in range(EPOCHS):
            start = time.time()
            gen_loss_avg = 0
            disc_loss_avg = 0
            steps = 0

            for image_batch, label_batch in dataset:
                g_loss, d_loss = train_step(image_batch, label_batch, generator, discriminator)
                gen_loss_avg += g_loss
                disc_loss_avg += d_loss
                steps += 1

            if steps > 0:
                gen_loss_avg /= steps
                disc_loss_avg /= steps

            print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec - Gen Loss: {gen_loss_avg:.4f}, Disc Loss: {disc_loss_avg:.4f}')

            if (epoch + 1) % 50 == 0:
                generate_and_save_images(generator, epoch + 1, seed_noise, seed_labels)

        # Save the final generator model
        model_save_path = os.path.join(SAVED_MODELS_DIR, 'cgan_generator.h5')
        generator.save(model_save_path)
        print(f"Final generator model saved to {model_save_path}")

        # Generate final batch of samples
        print("Generating final batch of samples...")
        final_save_path = generate_and_save_images(generator, EPOCHS, seed_noise, seed_labels, prefix='final_samples')

        print("\n" + "="*50)
        print(f"TRAINING COMPLETE. Generated images saved to:\n{GENERATED_SAMPLES_DIR}")
        print("="*50)
