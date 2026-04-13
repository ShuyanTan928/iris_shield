"""Iris-Shield — Reusable Streamlit UI components (v2 with verification panel)."""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from core.privacy_scorer import PrivacyReport


def render_header():
    st.markdown("""
    <div class="iris-header">
        <h1>Iris-Shield</h1>
        <p>Local-first AI privacy protection — your face, your identity, your control</p>
    </div>
    """, unsafe_allow_html=True)


def render_score_gauge(report: PrivacyReport):
    score = report.protected_score
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=score,
        delta={"reference": report.raw_score, "relative": False, "position": "bottom"},
        title={"text": "Privacy Protection Score", "font": {"size": 18}},
        number={"suffix": "%", "font": {"size": 42}},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": "#6c63ff"}, "bgcolor": "#1a1a2e",
            "steps": [
                {"range": [0, 40], "color": "#ff5252"},
                {"range": [40, 70], "color": "#ffab40"},
                {"range": [70, 100], "color": "#00e676"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig, use_container_width=True)


def render_detection_summary(report: PrivacyReport):
    c1, c2, c3 = st.columns(3)
    c1.metric("Faces", f"{report.faces_cloaked}/{report.faces_detected}", delta="cloaked")
    c2.metric("License Plates", f"{report.plates_replaced}/{report.plates_detected}", delta="replaced")
    c3.metric("Street Signs", f"{report.signs_removed}/{report.signs_detected}", delta="removed")


def render_before_after(original: np.ndarray, processed: np.ndarray):
    import cv2
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    c2.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Protected", use_container_width=True)


def render_pgd_details(cosine_distances: dict[str, float]):
    if not cosine_distances:
        return
    st.subheader("Adversarial Cloaking Details")
    for model, dist in cosine_distances.items():
        st.markdown(f"**{model}**: cosine distance = `{dist:.4f}`")
        st.progress(min(dist * 100, 100) / 100)


def render_pipeline_status(steps: dict[str, str]):
    css = {"done": "step-done", "running": "step-running", "pending": "step-pending"}
    icons = {"done": "OK", "running": ">>", "pending": ".."}
    badges = " ".join(f'<span class="pipeline-step {css.get(s, "step-pending")}">{icons.get(s, "..")} {n}</span>' for n, s in steps.items())
    st.markdown(badges, unsafe_allow_html=True)


def render_verification_panel(
    cosine_distances: dict[str, float],
    original_img: np.ndarray,
    protected_img: np.ndarray,
):
    """
    展示核心验证结果:
    - 每个模型的 similarity 从 ~1.0 降到了多少
    - 是否低于识别阈值
    - 对比柱状图
    """
    import cv2

    FR_THRESHOLD = 0.4  # 业界标准人脸识别匹配阈值

    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Identity Verification Test</h2>
        <p style="color: #aaa; margin: 0.3rem 0 0;">Does face recognition still identify the same person?</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 左右对比图 ──
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original Photo", use_container_width=True)
        st.markdown('<p style="text-align:center; color:#ff5252; font-weight:bold; font-size:1.2em;">VULNERABLE</p>', unsafe_allow_html=True)
    with col_img2:
        st.image(cv2.cvtColor(protected_img, cv2.COLOR_BGR2RGB), caption="Protected Photo", use_container_width=True)
        st.markdown('<p style="text-align:center; color:#00e676; font-weight:bold; font-size:1.2em;">PROTECTED</p>', unsafe_allow_html=True)

    # ── 每个模型的验证结果 ──
    st.markdown("### Per-Model Verification Results")
    st.markdown(f"**Recognition threshold: {FR_THRESHOLD}** (industry standard: similarity > {FR_THRESHOLD} = same person)")

    model_names = []
    before_scores = []
    after_scores = []
    all_below_threshold = True

    for model_name, cos_dist in cosine_distances.items():
        sim_before = 1.0  # original vs original
        sim_after = 1.0 - cos_dist  # original vs protected
        match_after = sim_after > FR_THRESHOLD

        model_names.append(model_name)
        before_scores.append(sim_before)
        after_scores.append(sim_after)

        if match_after:
            all_below_threshold = False

        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        with col1:
            st.markdown(f"**{model_name}**")
        with col2:
            st.markdown(f"Before: `{sim_before:.4f}`")
        with col3:
            st.markdown(f"After: `{sim_after:.4f}`")
        with col4:
            if not match_after:
                st.markdown(":white_check_mark: **NO MATCH**")
            else:
                st.markdown(":x: Still matched")

    # ── 对比柱状图 ──
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before Protection",
        x=model_names,
        y=before_scores,
        marker_color="#ff5252",
        text=[f"{s:.2f}" for s in before_scores],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="After Protection",
        x=model_names,
        y=after_scores,
        marker_color="#00e676",
        text=[f"{s:.4f}" for s in after_scores],
        textposition="outside",
    ))
    # Threshold line
    fig.add_hline(
        y=FR_THRESHOLD, line_dash="dash", line_color="yellow",
        annotation_text=f"Recognition Threshold ({FR_THRESHOLD})",
        annotation_position="top right",
        annotation_font_color="yellow",
    )
    fig.update_layout(
        title="Cosine Similarity: Before vs After Protection",
        yaxis_title="Cosine Similarity",
        yaxis_range=[0, 1.15],
        barmode="group",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 最终判定 ──
    if all_below_threshold:
        st.success(
            f"**PROTECTION SUCCESSFUL** — Identity cannot be matched by any of the "
            f"{len(cosine_distances)} recognition models tested. "
            f"All similarity scores dropped below the {FR_THRESHOLD} threshold."
        )
    else:
        st.warning(
            "**PARTIAL PROTECTION** — Some models still show similarity above threshold. "
            "Try increasing PGD steps or epsilon in the sidebar."
        )