import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.markdown("""
    <style>
    body {
        background-color: #1c1c1c;
        color: #dcdcdc;
    }
    .stButton>button {
        background-color: #3b3b3b;
        color: #f0f0f0;
        border-radius: 10px;
        border: 1px solid #565656;
    }
    .sidebar .sidebar-content {
        background-color: #252526;
        color: #dcdcdc;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FF9800;
    }
    .stRadio>div {
        background-color: #2d2d2d;
        color: #f0f0f0;
        border-radius: 10px;
    }
    .css-18e3th9 {
        color: #FF9800;
    }
    .block-container {
        padding: 4rem;  /* Increased padding to make the dashboard wider */
        background-color: #1c1c1c;
        border-radius: 10px;
        max-width: 1200px;  /* Max-width to control overall dashboard width */
        margin: auto;  /* Center dashboard */
        box-shadow: 0px 0px 15px 5px rgba(0,0,0,0.75);
    }
    .greeks-box {
        border: 1px solid #FF9800;
        padding: 10px;
        border-radius: 10px;
        background-color: #252526;
    }
    .greeks-box p {
        color: white;
        font-size: 16px;
        font-weight: bold;
    }
    .calculated-price {
        color: white;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Heston Model Option Pricing
def heston_option_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, N=10000, M=100):
    dt = T / M
    S = np.zeros((N, M + 1))
    v = np.zeros((N, M + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, M + 1):
        Z1 = np.random.normal(size=N)
        Z2 = np.random.normal(size=N)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # Correlated Brownian motion

        v[:, t] = np.maximum(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma_v * np.sqrt(v[:, t - 1]) * np.sqrt(dt) * W2, 0)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * np.sqrt(dt) * W1)

    payoff = np.maximum(S[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)

    return option_price

# Greeks Calculation
def compute_greeks(S0, K, T, r, kappa, theta, sigma_v, rho, v0):
    epsilon = 0.01  # Small perturbation for numerical derivatives
    option_price = heston_option_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0)

    delta = (heston_option_price(S0 + epsilon, K, T, r, kappa, theta, sigma_v, rho, v0) - option_price) / epsilon
    gamma = (heston_option_price(S0 + epsilon, K, T, r, kappa, theta, sigma_v, rho, v0) - 2 * option_price + heston_option_price(S0 - epsilon, K, T, r, kappa, theta, sigma_v, rho, v0)) / (epsilon ** 2)
    vega = (heston_option_price(S0, K, T, r, kappa, theta, sigma_v + epsilon, rho, v0) - option_price) / epsilon
    theta = (heston_option_price(S0, K, T - epsilon, r, kappa, theta, sigma_v, rho, v0) - option_price) / epsilon
    rho_val = (heston_option_price(S0, K, T, r + epsilon, kappa, theta, sigma_v, rho, v0) - option_price) / epsilon

    return delta, gamma, vega, theta, rho_val

# Volatility Surface Visualization
def plot_volatility_surface(S0, T, r, kappa, theta, sigma_v, rho, v0, figsize=(12,8)):
    strikes = np.linspace(60, 90, 10)
    maturities = np.linspace(0.1, 2, 5)
    vol_surface = np.zeros((len(strikes), len(maturities)))

    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            vol_surface[i, j] = heston_option_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0)

    X, Y = np.meshgrid(maturities, strikes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, vol_surface, cmap='plasma')  # Use a more striking colormap

    # Ensure all axis labels are correctly displayed
    ax.set_xlabel('Maturity (years)', color='white',labelpad=10)
    ax.set_ylabel('Strike Price', color='white',labelpad=10)
    ax.set_zlabel('Option Price', color='white',labelpad=0.1)
    
    plt.title("Volatility Surface", color="white")
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    fig.patch.set_facecolor('#2e2e2e')
    ax.set_facecolor('#2e2e2e')
    fig.tight_layout(pad=1.0)
    
    st.pyplot(fig)

# Streamlit App
st.title('Heston Model Interactive Dashboard')

# Input parameters via sliders in the sidebar
S0 = st.sidebar.slider("Initial Stock Price (Oil)", min_value=50.0, max_value=120.0, value=70.0)
K = st.sidebar.slider("Strike Price", min_value=60.0, max_value=100.0, value=75.0)
T = st.sidebar.slider("Time to Maturity (years)", min_value=0.1, max_value=2.0, value=1.0)
r = st.sidebar.slider("Risk-Free Rate", min_value=0.0, max_value=0.05, value=0.01)
kappa = st.sidebar.slider("Kappa (Speed of mean reversion)", min_value=0.1, max_value=5.0, value=2.0)
theta = st.sidebar.slider("Theta (Long-term volatility)", min_value=0.01, max_value=0.1, value=0.04)
sigma_v = st.sidebar.slider("Sigma (Volatility of volatility)", min_value=0.01, max_value=1.0, value=0.5)
rho = st.sidebar.slider("Rho (Correlation)", min_value=-1.0, max_value=1.0, value=-0.7)
v0 = st.sidebar.slider("Initial Volatility (v0)", min_value=0.01, max_value=0.1, value=0.04)

# Display Calculated Option Price with white font
st.markdown(f"<p class='calculated-price'>Calculated Call Option Price: ${heston_option_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0):.2f}</p>", unsafe_allow_html=True)

# Display Greeks inside a box with white font
delta, gamma, vega, theta, rho_val = compute_greeks(S0, K, T, r, kappa, theta, sigma_v, rho, v0)
st.markdown("""
    <div class='greeks-box'>
        <p>Delta: {:.4f}</p>
        <p>Gamma: {:.4f}</p>
        <p>Vega: {:.4f}</p>
        <p>Theta: {:.4f}</p>
        <p>Rho: {:.4f}</p>
    </div>
    """.format(delta, gamma, vega, theta, rho_val), unsafe_allow_html=True)

# Add heading and labels to the graphs
st.header("Visualizations")

# Display Volatility Surface with heading and labels
st.subheader("Volatility Surface")
plot_volatility_surface(S0, T, r, kappa, theta, sigma_v, rho, v0)
