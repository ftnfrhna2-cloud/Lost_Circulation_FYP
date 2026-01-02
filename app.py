import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set Page Config
st.set_page_config(page_title="Lost Circulation AI", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

# --- PAGE 1: HOME PAGE ---
if st.session_state.page == 'home':
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .header-text { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #FFFFFF; text-align: left; padding: 60px 0px 40px 0px; }
        .app-title { font-size: 70px; font-weight: 900; margin-bottom: 10px; white-space: nowrap; letter-spacing: -2px; color: #FFFFFF; text-align: left; }
        .app-subtitle { font-size: 26px; color: #FFA500; margin-top: 0px; font-weight: 400; letter-spacing: 1px; text-align: left; }
        
        .risk-management-text {
            font-size: 20px;
            line-height: 1.8;
            color: #d1d1d1;
            text-align: justify;
        }

        /* IMPROVED KEY FEATURES CARDS */
        .feature-card { 
            background-color: #1c1f26; 
            padding: 30px; /* Increased padding */
            border-radius: 12px; 
            text-align: left; 
            height: 260px; /* Increased height for stability */
            border-left: 6px solid #FFA500; 
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .feature-card h3 { 
            color: #FFA500; 
            font-size: 24px; /* Larger Title */
            margin-bottom: 15px; 
            font-weight: 800; 
        }
        .feature-card p { 
            font-size: 17px; /* Larger Body Text */
            color: #cccccc; 
            line-height: 1.6; 
        }
        
        div.stButton > button:first-child { background-color: #FFA500; color: white; font-weight: bold; border: none; width: 100%; height: 60px; font-size: 20px; border-radius: 10px; }
        div.stButton > button:last-child { background-color: transparent; color: #FFA500; border: 2px solid #FFA500; width: 100%; height: 60px; font-size: 20px; border-radius: 10px; }
        
        .footer { position: relative; margin-top: 100px; width: 100%; background-color: #0e1117; color: #888; text-align: center; padding: 30px; font-size: 14px; border-top: 1px solid #333; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="header-text"><h1 class="app-title">Lost Circulation Prediction System</h1><p class="app-subtitle">Machine Learning-Based Risk Prediction for Oil Drilling Operations</p></div>', unsafe_allow_html=True)

    st.write("---") 
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown("## Strategic Risk Management")
        st.markdown("""
        <div class="risk-management-text">
        Lost circulation remains a primary driver of <b>Non-Productive Time (NPT)</b> and excessive operational costs in modern drilling operations. 
        Traditional empirical models often fail to account for the complex, non-linear interactions between heterogeneous geological 
        formations and dynamic drilling parameters. By implementing an optimized XGBoost architecture, this system establishes a 
        proactive diagnostic framework, identifying high-risk zones and enabling engineers to implement real-time mitigation 
        strategies before bit penetration.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://www.linedpipesystems.com/wp-content/uploads/2023/01/on-site-oil-drilling.jpg", use_container_width=True)

    st.write("---") 
    
    # --- IMPROVED KEY FEATURES SECTION ---
    st.markdown("## Key Features")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown("""
            <div class="feature-card">
                <h3>Predictive Risk Analytics</h3>
                <p>Quantify subsurface circulation risks by analyzing multi-variant drilling parameters including ECD, hydrostatic pressures, and formation gradients.</p>
            </div>
        """, unsafe_allow_html=True)
    with f_col2:
        st.markdown("""
            <div class="feature-card">
                <h3>Advanced XGBoost Framework</h3>
                <p>Utilizes an optimized Gradient Boosting architecture trained on high-fidelity historical datasets to deliver superior predictive reliability.</p>
            </div>
        """, unsafe_allow_html=True)
    with f_col3:
        st.markdown("""
            <div class="feature-card">
                <h3>Operational Intelligence</h3>
                <p>Transform raw sensor data into actionable engineering mitigation protocols and real-time geomechanical safety window diagnostics.</p>
            </div>
        """, unsafe_allow_html=True)

    st.write("##")
    c_col1, c_col2, c_col3, c_col4 = st.columns([1, 1, 1, 1])
    with c_col2:
        if st.button("Start Prediction"):
            st.session_state.page = 'user_prediction'
            st.rerun()
    with c_col3:
        access_key = st.text_input("Developer Access", type="password", placeholder="Enter key")
        if access_key == "admin123":
            if st.button("Enter Developer Lab"):
                st.session_state.page = 'developer_lab'
                st.rerun()

    st.markdown("""
        <div class="footer">
            <p><b>Final Year Project (FYP)</b> | Bachelor of Chemical Engineering (Hons) Oil and Gas</p>
            <p>Developed by: <b>Fatin Farhana Farizman (2022940543)</b> | Supervised by: <b>Dr Nurul Aimi Ghazali</b></p>
        </div>
        """, unsafe_allow_html=True)

# --- PAGE 2: DEVELOPER LAB (UNTOUCHED) ---
elif st.session_state.page == 'developer_lab':
    st.title("Developer (Testing)")
    if st.button("← Exit to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    uploaded_file = st.file_uploader("Upload Historical CSV for Testing", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.fillna(df.median(numeric_only=True)).replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        le = LabelEncoder()
        df['FormationType'] = le.fit_transform(df['FormationType'].astype(str))
        st.session_state['le_final'] = le
        features = ['Depth_m', 'MudWeight_ppg', 'RotarySpeed_rpm', 'RateOfPenetration_mph', 'HoleDiameter_in', 'FormationType', 'FractureGradient_psi_per_ft', 'PumpPressure_psi', 'MudViscosity_cP', 'FluidLoss_mL_30min']
        X = df[features]; y = df['LostCirculation']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
        st.session_state['trained_model'] = model
        ml_acc = accuracy_score(y_test, model.predict(X_test))
        y_emp = X_test.apply(lambda r: 1 if (0.052 * r['MudWeight_ppg'] * r['Depth_m']) > (r['FractureGradient_psi_per_ft'] * r['Depth_m']) else 0, axis=1)
        emp_acc = accuracy_score(y_test, y_emp)
        
        st.success(f"Framework Ready! ML Accuracy: {round(ml_acc*100, 2)}% | Empirical: {round(emp_acc*100, 2)}%")
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(x=["Empirical", "XGBoost"], y=[emp_acc, ml_acc], color=["Trad", "AI"], color_discrete_sequence=["#555", "#FFA500"], template="plotly_dark", title="Accuracy Comparison"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance'), x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Oranges', template="plotly_dark", title="Feature Importance"), use_container_width=True)

# --- PAGE 3: USER PREDICTION (UNTOUCHED) ---
elif st.session_state.page == 'user_prediction':
    st.title("Lost Circulation Diagnostic System")
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ System Calibrating: Please contact the Developer to upload the training dataset.")
    else:
        st.info("💡 Adjust the sliders to see real-time risk changes and mitigation advice.")
        col_inp, col_viz = st.columns([1.2, 1.8], gap="large")
        with col_inp:
            st.markdown("### Operational Sliders")
            dm = st.slider("Target Depth (m)", 1000.0, 5000.0, 2500.0, step=10.0)
            mw = st.slider("Mud Weight (ppg)", 8.0, 18.0, 10.5, step=0.1)
            pp = st.slider("Pump Pressure (psi)", 1000, 4500, 2800, step=50)
            c1, c2 = st.columns(2)
            with c1:
                rpm, rop = st.number_input("Rotary Speed (rpm)", 50, 200, 110), st.number_input("ROP (m/hr)", 0.0, 60.0, 15.0)
                hd = st.number_input("Hole Dia (in)", 6.0, 24.0, 12.25)
            with c2:
                ft = st.selectbox("Formation", st.session_state['le_final'].classes_)
                fg, vis = st.number_input("FG (psi/ft)", 0.4, 1.2, 0.75), st.number_input("Viscosity (cP)", 10, 100, 45)
                fl = st.number_input("Fluid Loss", 0.0, 30.0, 5.0)

            ft_idx = st.session_state['le_final'].transform([ft])[0]
            prob = st.session_state['trained_model'].predict_proba(np.array([[dm, mw, rpm, rop, hd, ft_idx, fg, pp, vis, fl]]))[0][1]
            risk_pct = float(prob * 100)
            bhp, frac_lim = 0.052 * mw * dm, fg * dm
            margin = round(frac_lim - bhp, 0)
            st.divider()
            st.subheader("Borehole Assessment")
            m1, m2 = st.columns(2)
            m1.metric("Risk Probability", f"{risk_pct:.3f}%", delta=f"{risk_pct:.3f}%", delta_color="inverse")
            m2.metric("Safety Margin (psi)", f"{margin} psi", delta=f"{'SAFE' if margin > 0 else 'FRACTURE'}")
        with col_viz:
            st.markdown("### Predictive Depth Pattern")
            depth_axis = np.linspace(max(0, dm-500), dm+500, 200)
            risks = []
            for d in depth_axis:
                if abs(d - dm) < 70:
                    sim_prob = st.session_state['trained_model'].predict_proba(np.array([[d, mw, rpm, rop, hd, ft_idx, fg, pp, vis, fl]]))[0][1]
                    risks.append(sim_prob * np.exp(-((d - dm)**2) / (2 * 15**2)))
                else: risks.append(0)
            fig = go.Figure()
            fig.add_vrect(x0=0, x1=0.4, fillcolor="green", opacity=0.08, line_width=0)
            fig.add_vrect(x0=0.4, x1=0.8, fillcolor="yellow", opacity=0.08, line_width=0)
            fig.add_vrect(x0=0.8, x1=1.0, fillcolor="red", opacity=0.08, line_width=0)
            fig.add_trace(go.Scatter(x=risks, y=depth_axis, mode='lines', fill='tozerox', line=dict(color='#FFA500', width=4), name='Risk Spike'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[dm, dm], mode='lines', line=dict(color='white', width=2, dash='dash'), name='Bit Position'))
            fig.update_layout(xaxis=dict(title="Risk Level", range=[0, 1]), yaxis=dict(title="Measured Depth (m)", autorange="reversed"), template="plotly_dark", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Mitigation Action")
            if prob > 0.8: st.error("CRITICAL: Reduce Mud Weight immediately to restore safety margin.")
            elif prob > 0.4: st.warning("CAUTION: Operating at the geomechanical limit. Monitor pits closely.")
            else: st.success("STABLE: Operations within safe drilling window.")