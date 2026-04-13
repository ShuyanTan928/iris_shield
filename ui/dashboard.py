"""Iris-Shield dashboard — clean user-facing UI."""

from __future__ import annotations
import time
import streamlit as st
from PIL import Image
from loguru import logger
import config

from core.ensemble_cloaker import EnsembleCloaker


@st.cache_resource(show_spinner="Loading AI models (first time may take a minute)...")
def get_cloaker():
    return EnsembleCloaker()


def run_dashboard():
    # Header
    st.markdown("""
    <div class="iris-header">
        <h1>Iris-Shield</h1>
        <p>Protect your photos from AI crawlers, auto-labeling, and identity recognition.<br>
        Your image looks the same to humans, but AI sees something completely different.</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    uploaded = st.file_uploader(
        "Upload a photo to protect",
        type=["jpg", "jpeg", "png", "webp"],
        help="Your photo never leaves your computer. All processing happens locally.",
    )

    if uploaded is None:
        # Landing info
        st.markdown("### How it works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Upload**")
            st.markdown("Drop any photo with people, license plates, or private info.")
        with col2:
            st.markdown("**2. Protect**")
            st.markdown("Our AI adds invisible perturbations that fool vision models.")
        with col3:
            st.markdown("**3. Download**")
            st.markdown("Get back a photo that looks identical but is unreadable to AI.")
        return

    image = Image.open(uploaded).convert("RGB")

    # Resize if too large
    max_side = config.MAX_IMAGE_SIZE
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)

    st.image(image, caption="Your Photo", use_container_width=True)

    if not st.button("Protect My Photo", type="primary", use_container_width=True):
        return

    # Run attack with progress
    prog = st.progress(0, text="Loading AI models...")
    t0 = time.time()

    cloaker = get_cloaker()

    def update_progress(pct, msg):
        prog.progress(pct, text=msg)

    prog.progress(0.05, text="Analyzing your photo...")

    result = cloaker.cloak(image=image, progress_callback=update_progress)

    elapsed = time.time() - t0
    prog.progress(1.0, text=f"Done in {elapsed:.1f}s")

    # ── Results ──
    st.divider()

    # Before / After images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(result.cloaked_image, caption="Protected", use_container_width=True)

    # Caption comparison
    st.divider()
    st.markdown("### What AI sees")
    st.markdown("We asked an AI captioning model to describe both images:")

    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        st.markdown(f"""<div class="caption-box caption-before">
            <strong>Before protection:</strong><br>{result.caption_before}
        </div>""", unsafe_allow_html=True)
    with cap_col2:
        st.markdown(f"""<div class="caption-box caption-after">
            <strong>After protection:</strong><br>{result.caption_after}
        </div>""", unsafe_allow_html=True)

    # Background protection stats
    if result.plates_replaced > 0 or result.signs_blurred > 0:
        st.divider()
        st.markdown("### Background protection")
        bg_col1, bg_col2 = st.columns(2)
        with bg_col1:
            st.metric("License Plates Replaced", result.plates_replaced)
        with bg_col2:
            st.metric("Signs Blurred", result.signs_blurred)

    # Identity recognition comparison
    st.divider()
    st.markdown("### Identity recognition test")
    st.markdown("We tested whether AI can identify who is in the photo:")

    id_col1, id_col2 = st.columns(2)
    with id_col1:
        st.markdown("**Before protection:**")
        for name, score in result.identity_before:
            bar_pct = max(0, min(100, int(score * 100)))
            color = "#ff5252" if score > 0.25 else "#666"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
                f'<span>{name}</span><span style="color:{color};font-weight:bold">{score:.1%}</span></div>',
                unsafe_allow_html=True,
            )

    with id_col2:
        st.markdown("**After protection:**")
        for name, score in result.identity_after:
            bar_pct = max(0, min(100, int(score * 100)))
            color = "#00e676" if score < 0.2 else "#ffab40" if score < 0.25 else "#ff5252"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
                f'<span>{name}</span><span style="color:{color};font-weight:bold">{score:.1%}</span></div>',
                unsafe_allow_html=True,
            )

    # Top identity change
    if result.identity_before and result.identity_after:
        top_before = result.identity_before[0]
        top_after = result.identity_after[0]
        if top_before[0] != top_after[0] or top_after[1] < top_before[1] * 0.7:
            st.success(
                f"Identity recognition disrupted: "
                f"**{top_before[0]}** ({top_before[1]:.1%}) is no longer the top match."
            )

    # Summary
    st.divider()
    protection_keywords = ['redacted', 'protected', 'blocked', 'privacy', 'no visible', 'no content', 'data_redacted']
    caption_match = any(kw in result.caption_after.lower() for kw in protection_keywords)

    if caption_match:
        st.success(
            "### Your photo is protected\n\n"
            "AI crawlers and auto-labeling systems will no longer be able to correctly "
            "describe or identify the contents of this image. The photo looks identical "
            "to human eyes, but AI sees something completely different."
        )
    else:
        st.info(
            "### Partial protection applied\n\n"
            "The AI description has been altered, but may still contain some "
            "information about the original content. For stronger protection, "
            "try uploading a higher-resolution image."
        )

    # Download
    st.divider()
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(result.cloaked_image).save(buf, format="PNG")
    st.download_button(
        "Download Protected Photo",
        data=buf.getvalue(),
        file_name="iris_shield_protected.png",
        mime="image/png",
        use_container_width=True,
    )
