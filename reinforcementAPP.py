import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px

# --- SETUP ---
st.set_page_config(page_title="KI-Schulung: Reinforcement Learning", layout="wide")

# --- SESSION STATE INITIALISIERUNG ---
if 'q_table' not in st.session_state:
    st.session_state.q_table = np.zeros((25, 4))
if 'robot_pos' not in st.session_state:
    st.session_state.robot_pos = [0, 0]
if 'obstacles' not in st.session_state:
    st.session_state.obstacles = [[1, 1], [1, 2], [3, 2]]
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'last_log' not in st.session_state:
    st.session_state.last_log = "Bereit zum Lernen!"

# --- LOGIK ---
GRID_SIZE = 5
ACTIONS = ['Hoch', 'Runter', 'Links', 'Rechts']

def get_state(pos):
    return pos[0] * GRID_SIZE + pos[1]

def step(action_idx, alpha, gamma):
    r, c = st.session_state.robot_pos
    new_r, new_c = r, c
    
    if action_idx == 0: new_r -= 1
    elif action_idx == 1: new_r += 1
    elif action_idx == 2: new_c -= 1
    elif action_idx == 3: new_c += 1

    state = get_state([r, c])
    target = [4, 4]
    
    # Regeln & Belohnungen
    if not (0 <= new_r < GRID_SIZE and 0 <= new_c < GRID_SIZE):
        reward = -15 # Wand
        new_state = state
        msg = "💥 Wand kollidiert (-15)"
    elif [new_r, new_c] in st.session_state.obstacles:
        reward = -20 # Hindernis
        new_state = state
        msg = "🚫 Hindernis getroffen (-20)"
    elif [new_r, new_c] == target:
        reward = 100 # Ziel
        st.session_state.robot_pos = [new_r, new_c]
        new_state = get_state([new_r, new_c])
        msg = "🔋 Ladestation erreicht! (+100)"
    else:
        st.session_state.robot_pos = [new_r, new_c]
        new_state = get_state([new_r, new_c])
        reward = -1 # Jeder Schritt kostet Energie
        msg = "👣 Schritt gemacht (-1)"

    # Q-Learning Formel (Bellman-Gleichung vereinfacht)
    old_q = st.session_state.q_table[state, action_idx]
    max_future_q = np.max(st.session_state.q_table[new_state])
    
    # UPDATE
    new_q = old_q + alpha * (reward + gamma * max_future_q - old_q)
    st.session_state.q_table[state, action_idx] = new_q
    
    st.session_state.score += reward
    st.session_state.last_log = msg
    return reward

# --- UI ---
st.title("🤖 Interaktive Reinforcement Learning Demo")

# Sidebar für Parameter
st.sidebar.header("Konfiguration")
alpha = st.sidebar.slider("Lernrate (α)", 0.0, 1.0, 0.5, help="Wie stark beeinflussen neue Erfahrungen das alte Wissen?")
gamma = st.sidebar.slider("Diskontierung (γ)", 0.0, 1.0, 0.9, help="Wie wichtig sind zukünftige Belohnungen?")
epsilon = st.sidebar.slider("Zufallsrate (ε)", 0.0, 1.0, 0.2, help="Wahrscheinlichkeit für zufällige Erkundung")

if st.sidebar.button("Gehirn Reset"):
    st.session_state.q_table = np.zeros((25, 4))
    st.session_state.robot_pos = [0,0]
    st.session_state.score = 0
    st.rerun()

# --- Hauptbereich ---
col_map, col_brain = st.columns([1, 1])

with col_map:
    st.subheader("Spielfeld & Hindernisse")
    # Hindernis Management
    obs_options = [f"{r},{c}" for r in range(GRID_SIZE) for c in range(GRID_SIZE) if [r,c] != [0,0] and [r,c] != [4,4]]
    selected_obs = st.multiselect("Setze Hindernisse (🚫):", obs_options, 
                                  default=[f"{o[0]},{o[1]}" for o in st.session_state.obstacles])
    st.session_state.obstacles = [[int(x) for x in s.split(',')] for s in selected_obs]

    # Grid Visualisierung
    grid = np.full((GRID_SIZE, GRID_SIZE), "⬜")
    for o in st.session_state.obstacles: grid[o[0], o[1]] = "🚫"
    grid[4, 4] = "🔋"
    grid[st.session_state.robot_pos[0], st.session_state.robot_pos[1]] = "🤖"
    st.table(grid)
    st.write(f"**Status:** {st.session_state.last_log}")
    st.write(f"**Gesamt-Score:** {st.session_state.score}")

with col_brain:
    st.subheader("Das Wissen der KI (Q-Matrix)")
    # Heatmap der besten Aktionen pro Feld
    q_max = np.max(st.session_state.q_table, axis=1).reshape((5, 5))
    fig = px.imshow(q_max, text_auto=".1f", color_continuous_scale='RdYlGn', aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Die Zahlen zeigen den 'Wert' jedes Feldes. Grün = Erwartete Belohnung.")

# --- LERN-BUTTON ---
if st.button("10 Lernschritte machen"):
    for _ in range(10):
        if st.session_state.robot_pos == [4, 4]:
            time.sleep(1)
            st.session_state.robot_pos = [0, 0]
        
        state = get_state(st.session_state.robot_pos)
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(st.session_state.q_table[state])
        
        step(action, alpha, gamma)
        time.sleep(0.1)
    st.rerun()

# --- ERKLÄRUNG ---
st.divider()
st.subheader("🎓 Schulungs-Ecke: Wie lernt die KI?")
st.write("""
Der Roboter nutzt die **Bellman-Gleichung**. Jedes Mal, wenn er sich bewegt, aktualisiert er den Wert in seiner Matrix nach diesem Prinzip:
""")

st.latex(r"Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    **Was bedeuten die Werte?**
    * **Hohe positive Werte (Grün):** Die KI hat gelernt, dass sie von hier aus schnell zur Ladestation kommt.
    * **Negative Werte (Rot):** Die KI hat hier Schmerz erfahren (Kollision).
    * **Null (Weiß):** Dieses Feld oder diese Aktion ist noch völlig unbekannt.
    """)
with col_b:
    st.markdown("""
    **Der Lernprozess:**
    1.  **Trial & Error:** Am Anfang sind alle Werte 0. Er probiert zufällig Dinge aus.
    2.  **Belohnung:** Erreicht er die Station, wird der Weg dorthin "wertvoll".
    3.  **Propagierung:** Beim nächsten Mal weiß er schon kurz vor dem Ziel: "Hier war es gut!" So fließt das Wissen langsam vom Ziel zurück zum Start.
    """)
