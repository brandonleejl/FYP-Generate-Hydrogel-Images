import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- Configuration & Hyperparameters ---
IMG_SIZE = 256          # Image spatial dimension (256x256). Toggle to 512 if memory allows.
CHANNELS = 3            # RGB images
LATENT_DIM = 100        # Size of the random noise vector
BATCH_SIZE = 32         # Batch size for training
EPOCHS = 1000           # Total number of training epochs
LEARNING_RATE = 0.0002  # Adam learning rate
BETA_1 = 0.5            # Adam beta_1 parameter

# Paths (Google Drive Integration)
BASE_PATH = '/content/drive/MyDrive/FYP_Hydrogel_Data'
IMAGES_DIR = os.path.join(BASE_PATH, 'images')
LABELS_FILE = os.path.join(BASE_PATH, 'labels.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, 'generated_samples')

# Ensure output directory exists
# We wrap OS calls in try-except or checks to avoid errors if Drive isn't mounted locally
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError:
        print(f"Warning: Could not create directory {OUTPUT_DIR}. Check Drive mounting.")

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

    Args:
        path (tf.string): Path to the image file.
        label (tf.float32): The corresponding continuous pH label.

    Returns:
        img (tf.Tensor): The processed image tensor.
        label (tf.Tensor): The label tensor.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS) # Adjust if PNG
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    # Normalize to [-1, 1] for Tanh activation in Generator
    img = (img - 127.5) / 127.5

    return img, label

def augment_image(img, label):
    """
    Applies random horizontal and vertical flips to the image for data augmentation.
    This helps prevent overfitting by introducing variance in the training data.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, label

def create_dataset():
    """
    Creates a tf.data.Dataset from the CSV file and image directory.

    Returns:
        dataset (tf.data.Dataset): A batched and prefetched dataset ready for training.
    """
    # Check if files exist (for local testing without Drive)
    if not os.path.exists(LABELS_FILE):
        print(f"Error: Labels file not found at {LABELS_FILE}")
        return None

    # Read CSV
    df = pd.read_csv(LABELS_FILE)

    # Construct full image paths
    # Assuming 'filename' column exists and contains names like 'image_01.jpg'
    image_paths = [os.path.join(IMAGES_DIR, fname) for fname in df['filename']]
    labels = df['ph'].values.astype(np.float32).reshape(-1, 1) # Reshape to (N, 1) for Keras input compatibility

    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Map loading and preprocessing functions
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.shuffle(buffer_size=len(df)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset

# --- Model Architecture: Generator ---

def build_generator():
    """
    Builds the Conditional Generator model.

    Architecture:
    - Inputs: Latent noise vector (z) and continuous pH label (c).
    - The label is processed via a Dense layer to embed it into a feature space.
    - Concatenation: The processed label and noise are concatenated.
    - Upsampling: Uses Conv2DTranspose layers with BatchNorm and ReLU to progressively
      increase spatial resolution.
    - Output: Tanh activation to produce images in range [-1, 1].
    """
    # Inputs
    noise_input = Input(shape=(LATENT_DIM,), name='noise_input')
    label_input = Input(shape=(1,), name='label_input') # Single continuous float

    # --- Conditioning Strategy ---
    # Since the label is continuous, we project it into a higher dimensional space
    # using a Dense layer so it carries enough weight when combined with the noise.
    # We aim to reshape the inputs to the starting spatial dimension of the generator (e.g., 4x4 or 8x8).

    start_dim = IMG_SIZE // 32 # E.g., 256 / 32 = 8
    n_nodes = start_dim * start_dim * 256

    # Process Label
    label_embedding = layers.Dense(start_dim * start_dim * 1)(label_input)
    label_embedding = layers.Reshape((start_dim, start_dim, 1))(label_embedding)

    # Process Noise
    noise_embedding = layers.Dense(start_dim * start_dim * 256)(noise_input)
    noise_embedding = layers.Reshape((start_dim, start_dim, 256))(noise_embedding)

    # Concatenate: Channel-wise concatenation of noise feature map and label feature map
    x = layers.Concatenate()([noise_embedding, label_embedding])

    # --- Upsampling Blocks ---
    # Block 1: 8x8 -> 16x16
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2: 16x16 -> 32x32
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 3: 32x32 -> 64x64
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 4: 64x64 -> 128x128
    x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 5: 128x128 -> 256x256
    x = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    output = layers.Activation('tanh')(x) # Output range [-1, 1]

    model = models.Model([noise_input, label_input], output, name="Generator")
    return model

# --- Model Architecture: Discriminator ---

def build_discriminator():
    """
    Builds the Conditional Discriminator model.

    Architecture:
    - Inputs: Image (real or fake) and continuous pH label.
    - Conditioning: The label is spatially replicated to match the image dimensions
      and concatenated as an extra channel (RGB + Label).
    - Downsampling: Uses Conv2D layers with LeakyReLU and Dropout.
    - Output: Single scalar (validity score).
    """
    # Inputs
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name='image_input')
    label_input = Input(shape=(1,), name='label_input')

    # --- Conditioning Strategy ---
    # Specifically for continuous labels, we project the scalar to a feature map
    # matching the image resolution (IMG_SIZE x IMG_SIZE).
    # This creates a "label plane" that explicitly tells the discriminator
    # the target pH for every pixel location.

    # Option 1: Spatially replicate the scalar.
    # Option 2: Dense layer followed by Reshape (more flexible, allows learning).
    # We'll use a Dense layer to allow the network to learn a representation of the pH
    # suitable for concatenation, reshaped to (IMG_SIZE, IMG_SIZE, 1).

    # As requested for the FYP report architecture:
    # Process the continuous float label through a Dense layer and reshape it
    # to match the spatial dimensions of the image.

    label_embedding = layers.Dense(IMG_SIZE * IMG_SIZE)(label_input)
    label_channel = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(label_embedding)

    # Concatenate: (IMG_SIZE, IMG_SIZE, 3) + (IMG_SIZE, IMG_SIZE, 1) -> (IMG_SIZE, IMG_SIZE, 4)
    x = layers.Concatenate()([img_input, label_channel])

    # --- Downsampling Blocks ---
    # Block 1
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x) # Logits output (no sigmoid here, use from_logits=True in loss)

    model = models.Model([img_input, label_input], output, name="Discriminator")
    return model

# --- Loss Functions & Optimizers ---

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """
    Discriminator Loss:
    - Should correctly classify real images as 1 (real_loss).
    - Should correctly classify fake images as 0 (fake_loss).
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    Generator Loss:
    - Tries to fool the discriminator into classifying fake images as 1.
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

# --- Training Loop ---

@tf.function
def train_step(images, labels, generator, discriminator):
    """
    Performs one training step.
    """
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images conditioned on the labels
        generated_images = generator([noise, labels], training=True)

        # Discriminator pass
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, test_labels):
    """
    Generates a grid of images for visualization and saves to Drive.
    """
    predictions = model([test_input, test_labels], training=False)

    fig = plt.figure(figsize=(10, 10))

    # Rescale to [0, 1] for display
    predictions = (predictions + 1) / 2.0

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.title(f"pH: {test_labels[i, 0]:.1f}")
        plt.axis('off')

    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(save_path)
    plt.close() # Close to free memory
    print(f"Saved sample grid to {save_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()

    print("Models initialized.")
    generator.summary()
    discriminator.summary()

    # Load dataset
    dataset = create_dataset()

    if dataset is None:
        print("Dataset creation failed. Exiting training setup.")
    else:
        # Seed for consistent visualization
        num_examples_to_generate = 16
        seed_noise = tf.random.normal([num_examples_to_generate, LATENT_DIM])
        # Generate random pH labels for visualization (e.g., between 4.0 and 8.0)
        seed_labels = np.random.uniform(4.0, 8.0, (num_examples_to_generate, 1)).astype(np.float32)

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

            print ('Time for epoch {} is {} sec - Gen Loss: {:.4f}, Disc Loss: {:.4f}'.format(epoch + 1, time.time()-start, gen_loss_avg, disc_loss_avg))

            # Save generated images every 50 epochs
            if (epoch + 1) % 50 == 0:
                generate_and_save_images(generator, epoch + 1, seed_noise, seed_labels)

        # Save the final model
        model_save_path = os.path.join(BASE_PATH, 'cgan_generator.h5')
        generator.save(model_save_path)
        print(f"Final generator model saved to {model_save_path}")
