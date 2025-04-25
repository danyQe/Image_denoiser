import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from keras_hub.src.models.resnet import resnet_backbone
import time
import logging
import traceback
import kagglehub

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom objects and constants
CUSTOM_OBJECTS = {'ResNetBackbone': resnet_backbone.ResNetBackbone}
NOISE_TYPES = ['speckle', 'jpeg', 'multiplicative', 'gaussian', 'salt_and_pepper', 'quantization', 'poisson']
IMG_SIZE = 256
DEFAULT_PATCH_SIZE = 80
DEFAULT_OVERLAP = 20
# Configure TensorFlow memory settings
def configure_gpu():
    """Configure GPU memory growth to prevent OOM errors"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    else:
        logger.info("No GPUs found, using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False

# Download latest version
path = kagglehub.model_download("goutham1208/image-denoiser/tensorFlow2/default")

print("Path to model files:", path)
@st.cache_resource
def load_noise_classifier():
    """Load the noise classification model with caching"""
    logger.info("Loading noise classifier model")
    try:
        model = load_model(f'{path}/noise_classifier.keras', custom_objects=CUSTOM_OBJECTS, compile=False)
        logger.info("Noise classifier loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load noise classifier: {e}")
        raise

def load_denoiser(model_path):
    """Load a specific denoiser model"""
    logger.info(f"Loading denoiser model: {model_path}")
    try:
        model = load_model(model_path, compile=False)
        logger.info(f"Denoiser model loaded successfully: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load denoiser model {model_path}: {e}")
        raise

def preprocess_image(img):
    """Preprocess image for model input"""
    try:
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype('float32') / 255.0
        return img_rgb, np.expand_dims(img_norm, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def split_into_patches(img, patch_size=DEFAULT_PATCH_SIZE, overlap=DEFAULT_OVERLAP):
    """Split image into overlapping patches for processing"""
    h, w, _ = img.shape
    stride = patch_size - overlap
    
    # Calculate padded dimensions
    padded_h = ((h - 1) // stride + 1) * stride + overlap
    padded_w = ((w - 1) // stride + 1) * stride + overlap
    
    # Create padded image
    padded_img = cv2.copyMakeBorder(
        img, 
        0, max(0, padded_h - h), 
        0, max(0, padded_w - w), 
        cv2.BORDER_REFLECT_101
    )
    
    # Extract patches
    patches, positions = [], []
    for y in range(0, padded_h - patch_size + 1, stride):
        for x in range(0, padded_w - patch_size + 1, stride):
            patches.append(padded_img[y:y+patch_size, x:x+patch_size])
            positions.append((y, x))
    
    logger.debug(f"Split image into {len(patches)} patches")
    return np.array(patches), positions, (h, w, padded_h, padded_w)

def reconstruct_from_patches(patches, positions, shape, patch_size=DEFAULT_PATCH_SIZE, overlap=DEFAULT_OVERLAP):
    """Reconstruct image from processed patches"""
    h, w, padded_h, padded_w = shape
    reconstructed = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    weight_mask = np.ones((patch_size, patch_size, 1), dtype=np.float32)
    
    # Create weight mask for smooth blending at overlaps
    if overlap > 0:
        for i in range(overlap):
            # Linear weight reduction at edges
            factor = (i + 1) / (overlap + 1)
            weight_mask[i, :, 0] *= factor
            weight_mask[patch_size-i-1, :, 0] *= factor
            weight_mask[:, i, 0] *= factor
            weight_mask[:, patch_size-i-1, 0] *= factor
    
    # Apply patches with weights
    counts = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    for patch, (y, x) in zip(patches, positions):
        patch_weighted = patch * weight_mask
        reconstructed[y:y+patch_size, x:x+patch_size] += patch_weighted
        counts[y:y+patch_size, x:x+patch_size] += weight_mask
    
    # Normalize by weight counts
    counts = np.maximum(counts, 1e-10)  # Avoid division by zero
    final_img = (reconstructed / counts).astype(np.uint8)
    
    # Return original image dimensions
    return final_img[:h, :w]

def denoise_large_image(img, model, patch_size=DEFAULT_PATCH_SIZE, progress_bar=None):
    """Process large images by splitting into patches, denoising, and reconstructing"""
    start_time = time.time()
    
    # Ensure image is RGB
    if img.ndim == 2:
        logger.debug("Converting grayscale image to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        logger.debug("Converting RGBA image to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Split image into patches
    patches, positions, shape = split_into_patches(img, patch_size)
    patches_input = patches.astype(np.float32) / 255.0
    
    # Process patches in batches
    batch_size = 4
    denoised_patches = []
    
    total_batches = (len(patches_input) + batch_size - 1) // batch_size
    for i in range(0, len(patches_input), batch_size):
        batch_start = i
        batch_end = min(i + batch_size, len(patches_input))
        
        if progress_bar:
            progress_value = min((i + batch_size) / len(patches_input), 1.0)
            progress_bar.progress(progress_value, f"Processing batch {(i//batch_size)+1}/{total_batches}")
            
        batch_result = model.predict(patches_input[batch_start:batch_end], verbose=0)
        denoised_patches.append(batch_result)

    # Combine batch results
    denoised_patches = np.vstack(denoised_patches)
    denoised_patches = (np.clip(denoised_patches, 0, 1) * 255).astype(np.uint8)
    
    # Reconstruct final image
    reconstructed = reconstruct_from_patches(denoised_patches, positions, shape, patch_size)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Image denoising completed in {elapsed_time:.2f} seconds")

    return reconstructed

def calculate_metrics(original, denoised):
    """Calculate PSNR and SSIM metrics between original and denoised images"""
    try:
        # Ensure images are same size
        if original.shape[:2] != denoised.shape[:2]:
            denoised = cv2.resize(denoised, (original.shape[1], original.shape[0]))
        
        # Calculate PSNR
        ps = psnr(original, denoised)
        
        # Calculate SSIM (try different window sizes if needed)
        try:
            ss = ssim(original, denoised, channel_axis=-1)
        except ValueError:
            ss = ssim(original, denoised, channel_axis=-1, win_size=3)
            
        logger.debug(f"Metrics - PSNR: {ps:.2f}, SSIM: {ss:.2f}")
        return ps, ss
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return 0.0, 0.0

def denoise_pipeline(img, manual_noises=None, progress_placeholder=None, threshold=0.05):
    """Main denoising pipeline that applies appropriate denoisers and combines results"""
    start_time = time.time()
    denoised_outputs = []
    weights = []
    psnr_scores = {}
    ssim_scores = {}
    
    # Determine which noise types to process
    if manual_noises:
        logger.info(f"Using manually selected noise types: {manual_noises}")
        present_noises = manual_noises
        noise_info = {n: None for n in present_noises}
    else:
        # Use classifier to detect noise types
        logger.info("Running noise classifier")
        noise_classifier = load_noise_classifier()
        img_rgb, img_input = preprocess_image(img)
        
        preds = noise_classifier.predict(img_input, verbose=0)[0]
        present_indices = np.argsort(preds)[::-1]  # Reverse to get highest confidence first
        present_noises = [NOISE_TYPES[i] for i in present_indices]
        noise_info = {NOISE_TYPES[i]: float(preds[i]) for i in present_indices}
        
        logger.info(f"Classifier predictions: {noise_info}")
    
    # Apply denoisers for each detected/selected noise type
    for noise, prob in noise_info.items():
        if prob is None or prob >= threshold:  # Process manual selections or high confidence detections
            try:
                model_path = os.path.join(path, f'ridnet_{noise}.keras')
                if not os.path.exists(model_path):
                    logger.warning(f"Model not found: {model_path}")
                    continue
                    
                model = load_denoiser(model_path)
                
                if progress_placeholder:
                    progress_placeholder.text(f"Denoising for: {noise}")
                    progress_bar = st.progress(0.0)
                else:
                    progress_bar = None
                
                logger.info(f"Processing {noise} noise")
                denoised = denoise_large_image(img, model, patch_size=DEFAULT_PATCH_SIZE, progress_bar=progress_bar)
                denoised_outputs.append(denoised.astype(np.float32))
                weights.append(prob if prob is not None else 1.0)
                
                # Calculate metrics if we have a valid probability
                if prob is not None and prob > threshold:
                    psnr_val, ssim_val = calculate_metrics(img, denoised)
                    psnr_scores[noise] = psnr_val
                    ssim_scores[noise] = ssim_val
            except Exception as e:
                logger.error(f"Error processing {noise} noise: {e}")
    
    # If no denoisers were applied, return original image
    if not denoised_outputs:
        logger.warning("No denoising performed - returning original image")
        return img, img, noise_info, {}
    
    # Calculate weighted sum of all denoiser outputs
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()  # Normalize weights
    
    fused = sum(wi * out for wi, out in zip(w, denoised_outputs))
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Complete denoising pipeline finished in {elapsed_time:.2f} seconds")
    
    return img, fused, noise_info, psnr_scores

def show_prediction_graph(predictions):
    """Generate and display a bar chart of noise predictions"""
    # Safety checks for empty predictions or None
    if not predictions:
        st.warning("No prediction data available to display")
        return
    
    # Filter out None values and ensure we have numeric values
    filtered_predictions = {}
    for k, v in predictions.items():
        if v is not None and isinstance(v, (int, float)):
            filtered_predictions[k] = v
    
    if not filtered_predictions:
        st.warning("No valid numeric prediction data available to display")
        return
    
    # Now create the plot with valid data
    plt.figure(figsize=(10, 5))
    
    noises = list(filtered_predictions.keys())
    probs = list(filtered_predictions.values())
    
    # Another safety check
    if not noises or not probs:
        st.warning("Empty data for prediction graph")
        return
    
    # Sort by probability
    sorted_data = sorted(zip(noises, probs), key=lambda x: x[1], reverse=True)
    sorted_noises = [x[0] for x in sorted_data]
    sorted_probs = [x[1] for x in sorted_data]
    
    # Create the bar chart
    try:
        bars = plt.bar(sorted_noises, sorted_probs, color='skyblue')
        plt.ylabel("Confidence Score", fontsize=12)
        plt.xlabel("Noise Type", fontsize=12)
        plt.ylim(0, 1)
        plt.title("Noise Type Prediction Confidence", fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar, prob in zip(bars, sorted_probs):
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 0.01, 
                f"{prob:.2f}", 
                ha='center',
                va='bottom'
            )
        plt.tight_layout()
        return plt
    except Exception as e:
        logger.error(f"Error creating prediction graph: {e}")
        st.error(f"Could not create prediction graph: {str(e)}")
        return None

def display_noise_predictions(predictions):
    """Alternative way to display predictions using Streamlit native components"""
    # Filter valid predictions
    valid_predictions = {k: v for k, v in predictions.items() if v is not None and isinstance(v, (int, float))}
    
    if not valid_predictions:
        st.warning("No valid prediction data available")
        return
    
    # Sort by confidence
    sorted_items = sorted(valid_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Create dataframe
    df = pd.DataFrame({
        'Noise Type': [item[0] for item in sorted_items],
        'Confidence': [item[1] for item in sorted_items]
    })
    
    # Display as chart
    st.bar_chart(df.set_index('Noise Type'))
    
    # Display as table
    st.dataframe(df, hide_index=True)

def debug_classifier_on_test_images(sample_limit=3):
    """Test the classifier on images with known noise types"""
    logger.info("Running classifier debug on test images")
    noise_classifier = load_noise_classifier()
    
    results = {}
    
    # Create columns for display
    cols = st.columns(len(NOISE_TYPES))
    for idx, noise_type in enumerate(NOISE_TYPES):
        cols[idx].subheader(f"{noise_type}")
    
    for noise_type in NOISE_TYPES:
        test_dir = f"dataset/noises/{noise_type}"
        if not os.path.exists(test_dir):
            logger.warning(f"Test directory not found: {test_dir}")
            continue
            
        img_files = os.listdir(test_dir)[:sample_limit]
        
        for img_file in img_files:
            img_path = os.path.join(test_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            # Test the classifier
            _, img_input = preprocess_image(img)
            preds = noise_classifier.predict(img_input, verbose=0)[0]
            
            # Store results
            predicted_noise = NOISE_TYPES[np.argmax(preds)]
            confidence = preds[np.argmax(preds)]
            
            results[img_file] = {
                'true_noise': noise_type,
                'predicted_noise': predicted_noise,
                'confidence': confidence,
                'all_predictions': {n: float(p) for n, p in zip(NOISE_TYPES, preds)}
            }
            
            # Display sample in appropriate column
            col_idx = NOISE_TYPES.index(noise_type)
            cols[col_idx].image(
                img, 
                caption=f"Pred: {predicted_noise} ({confidence:.2f})",
                width=150
            )
    
    return results

def create_ui():
    """Create the Streamlit UI layout and components"""
    # Configure page
    st.set_page_config(
        page_title="Advanced Image Denoising",
        page_icon="🖼️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #1E88E5 !important;
            text-align: center !important;
            margin-bottom: 1.5rem !important;
        }
        .sub-header {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            color: #43a047 !important;
            margin-top: 1rem !important;
        }
        .stProgress > div > div {
            background-color: #1E88E5 !important;
        }
        .highlight-box {
            background-color: #f0f8ff;
            border-radius: 0.5rem;
            padding: 1rem;
            border-left: 0.5rem solid #1E88E5;
        }
        .metric-value {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            color: #1E88E5 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App Header
    st.markdown("<h1 class='main-header'>🖼️ Advanced Image Denoising App</h1>", unsafe_allow_html=True)
    
    # App description
    with st.expander("ℹ️ About This App", expanded=False):
        st.markdown("""
        This application uses deep learning models to detect and remove various types of noise from images:
        
        - **Gaussian Noise**: Random variation in brightness or color
        - **Speckle Noise**: Multiplicative noise commonly seen in radar images
        - **Salt & Pepper Noise**: Random black and white pixels
        - **Poisson Noise**: Shot noise from photon counting
        - **Multiplicative Noise**: Signal-dependent noise
        - **JPEG Compression Artifacts**: Blocky artifacts from JPEG compression
        - **Quantization Noise**: Errors from converting continuous values to discrete levels
        
        The app can automatically detect noise or allow manual selection.
        """)
    
    return st.sidebar

def main():
    """Main application function"""
    # Initialize GPU configuration
    has_gpu = configure_gpu()
    
    # Create UI
    sidebar = create_ui()
    
    # Sidebar settings
    sidebar.header("⚙️ Settings")
    
    threshold = sidebar.slider(
        "Noise Detection Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Minimum confidence needed to apply a denoiser"
    )
    
    debug_mode = sidebar.checkbox("🔍 Debug Mode", help="Show raw classifier predictions")
    
    advanced_mode = sidebar.checkbox("🛠️ Advanced Mode", help="Enable advanced options")
    
    if advanced_mode:
        patch_size = sidebar.slider(
            "Patch Size",
            min_value=32,
            max_value=128,
            value=DEFAULT_PATCH_SIZE,
            step=16,
            help="Size of image patches for processing"
        )
        
        overlap = sidebar.slider(
            "Patch Overlap",
            min_value=0,
            max_value=40,
            value=DEFAULT_OVERLAP,
            step=4,
            help="Overlap between adjacent patches"
        )
    else:
        patch_size = DEFAULT_PATCH_SIZE
        overlap = DEFAULT_OVERLAP
    
    # Model validation
    if not os.path.exists("models"):
        st.error("❌ Missing 'models' directory. Please ensure model files are available.")
        st.stop()
    
    # System info display
    if has_gpu:
        sidebar.success("✅ Using GPU acceleration")
    else:
        sidebar.warning("⚠️ Running on CPU")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image to denoise...",
        type=["jpg", "jpeg", "png"],
        help="Upload a noisy image you want to clean"
    )
    
    # Stop if no file uploaded
    if not uploaded_file:
        st.info("👆 Please upload an image to get started")
        
        # # Show sample images
        # st.markdown("<h2 class='sub-header'>Sample Results</h2>", unsafe_allow_html=True)
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.image("https://via.placeholder.com/400x300.png?text=Noisy+Image", caption="Before")
        # with col2:
        #     st.image("https://via.placeholder.com/400x300.png?text=Denoised+Image", caption="After")
        return
    
    # Process uploaded image
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("❌ Failed to read the uploaded image")
            return
            
        # Display original image
        st.markdown("<h2 class='sub-header'>Original Image</h2>", unsafe_allow_html=True)
        st.image(img)
        
        # Noise selection options
        col1, col2 = st.columns([2, 1])
        with col1:
            manual = st.checkbox("Manual noise selection", help="Manually select which noise types to process")
        
        selected = None
        if manual:
            with col2:
                selected = st.multiselect("Select noise types to process:", NOISE_TYPES)
        
        # Denoising button
        if st.button("🔄 Denoise Image", type="primary", use_container_width=True):
            # Progress placeholder
            progress_container = st.container()
            placeholder = progress_container.empty()
            placeholder.text("Starting denoising process...")
            
            # Run classifier in debug mode if selected
            if debug_mode and not manual:
                with st.expander("🔍 Raw Noise Classifier Output", expanded=True):
                    st.info("Analyzing image for noise types...")
                    
                    # Get raw predictions
                    img_norm, img_input = preprocess_image(img)
                    noise_classifier = load_noise_classifier()
                    preds = noise_classifier.predict(img_norm, verbose=0)[0]
                    
                    # Create prediction dict and display using streamlit native components
                    pred_dict = {noise_type: float(preds[i]) for i, noise_type in enumerate(NOISE_TYPES)}
                    display_noise_predictions(pred_dict)
            
            # Run denoising pipeline
            try:
                start_time = time.time()
                orig, denoised, info, psnr_scores = denoise_pipeline(
                    img, 
                    manual_noises=selected, 
                    progress_placeholder=placeholder,
                    threshold=threshold
                )
                elapsed_time = time.time() - start_time
                
                # Clear progress and show results
                progress_container.empty()
                st.success(f"✅ Denoising completed in {elapsed_time:.2f} seconds!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h3 class='sub-header'>Original</h3>", unsafe_allow_html=True)
                    st.image(orig, use_container_width=True)
                
                with col2:
                    st.markdown("<h3 class='sub-header'>Denoised</h3>", unsafe_allow_html=True)
                    st.image(denoised, use_container_width=True)
                    
                    # Provide download button for denoised image
                    denoised_bytes = cv2.imencode('.png', denoised)[1].tobytes()
                    st.download_button(
                        label="💾 Download Denoised Image",
                        data=denoised_bytes,
                        file_name="denoised.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Show metrics and details in expandable sections
                with st.expander("📊 Quality Metrics", expanded=True):
                    ps_val, ss_val = calculate_metrics(orig, denoised)
                    
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.markdown("""
                        <div style="background-color: #ffffff;color: #000000; border-radius: 0.5rem; padding: 1rem; border-left: 0.5rem solid #1E88E5;" class="highlight-box">
                            <h4>PSNR (Peak Signal-to-Noise Ratio)</h4>
                            <p>Higher is better, typically 20-40 dB is good</p>
                            <p class="metric-value">{:.2f} dB</p>
                        </div>
                        """.format(ps_val), unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown("""
                            <div style="background-color: #ffffff;color: #000000; border-radius: 0.5rem; padding: 1rem; border-left: 0.5rem solid #1E88E5;" class="highlight-box">
                            <h4>SSIM (Structural Similarity Index)</h4>
                            <p>Closer to 1.0 is better</p>
                            <p class="metric-value">{:.3f}</p>
                        </div>
                        """.format(ss_val), unsafe_allow_html=True)
                    
                    # Show per-denoiser metrics
                    if psnr_scores:
                        st.markdown("### PSNR Scores for Each Denoiser")
                        
                        # Create dataframe for metrics
                        metrics_df = pd.DataFrame({
                            'Noise Type': list(psnr_scores.keys()),
                            'PSNR (dB)': [f"{score:.2f}" for score in psnr_scores.values()]
                        })
                        
                        # Display as a sortable table
                        st.dataframe(
                            metrics_df.sort_values('PSNR (dB)', ascending=False),
                            hide_index=True,
                            use_container_width=True
                        )
                
                with st.expander("🔍 Applied Noise Processing", expanded=True):
                    st.markdown("### Detected/Applied Noise Types")
                    
                    # Create and display processed noise info
                    noise_data = []
                    for noise_type, prob in info.items():
                        if prob is None:  # Manual selection
                            status = "✅ Applied (Manual)"
                            confidence = "N/A"
                        elif prob >= threshold:  # Auto detected and applied
                            status = "✅ Applied (Auto)"
                            confidence = f"{prob:.4f}"
                        else:  # Detected but not applied
                            status = "❌ Not Applied"
                            confidence = f"{prob:.4f}"
                            
                        noise_data.append({
                            "Noise Type": noise_type,
                            "Confidence": confidence,
                            "Status": status
                        })
                    
                    # Display as dataframe
                    noise_df = pd.DataFrame(noise_data)
                    st.dataframe(noise_df, hide_index=True, use_container_width=True)
                
                # Option to compare with advanced debugging
                if advanced_mode:
                    with st.expander("🔬 Advanced Analysis", expanded=False):
                        st.markdown("### Image Difference Analysis")
                        
                        # Generate difference image
                        if orig.shape == denoised.shape:
                            diff_img = cv2.absdiff(orig, denoised)
                            # Enhance difference for visibility
                            diff_img = cv2.convertScaleAbs(diff_img, alpha=5.0)
                            
                            st.image(diff_img, caption="Enhanced Difference (5x amplified)", use_container_width=True)
                            
                            # Show histogram of differences
                            st.markdown("### Difference Histogram")
                            fig, ax = plt.subplots(figsize=(10, 4))
                            
                            # Calculate raw difference
                            raw_diff = cv2.absdiff(orig, denoised)
                            
                            # Plot histogram for each channel
                            colors = ['b', 'g', 'r']
                            for i, color in enumerate(colors):
                                hist = cv2.calcHist([raw_diff], [i], None, [256], [0, 256])
                                ax.plot(hist, color=color, label=f"Channel {i}")
                            
                            ax.set_xlabel('Pixel Difference Value')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Histogram of Differences Between Original and Denoised Image')
                            ax.legend()
                            st.pyplot(fig)
            
            except Exception as e:
                logger.error(f"Error in denoising pipeline: {e}")
                st.error(f"❌ Error during denoising: {str(e)}")
                st.code(traceback.format_exc())
            
            finally:
                placeholder.empty()
        
        # Expert mode - classifier testing
        if debug_mode and advanced_mode and st.button("📋 Run Classifier Tests"):
            st.markdown("<h2 class='sub-header'>Classifier Test Results</h2>", unsafe_allow_html=True)
            
            try:
                test_results = debug_classifier_on_test_images()
                
                # Display confusion matrix if there are results
                if test_results:
                    # Make a summary of results
                    correct = sum(1 for r in test_results.values() if r['true_noise'] == r['predicted_noise'])
                    total = len(test_results)
                    accuracy = correct / total if total > 0 else 0
                    
                    st.markdown(f"### Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
                else:
                    st.warning("No test images found in the dataset directory")
            except Exception as e:
                st.error(f"Error running classifier tests: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in main app function: {e}")
        st.error(f"❌ An error occurred: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()