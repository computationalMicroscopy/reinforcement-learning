import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px

# --- SETUP ---
st.set_page_config(page_title="KI-Trainer: RL Saugroboter", layout="wide")

# CSS für schönere Darstellung der Grids
st.markdown("""
    <style>
    .grid-cell { height: 60px; display: flex; align-items: center; justify-content: center; border: 1px solid #ddd; font-size: 25px; }
    .status-box { padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIK-KLASSE FÜR RL ---
class RLSimulator:
    def __init__(self, size=5):
        self.size = size
        self.actions = [0, 1, 2, 3] # hoch, runter, links, rechts
        self.reset()

    def reset(self):
        self.robot_pos = [0, 0]
        self.target_pos = [4, 4]
        self.obstacles = [[1, 1], [1, 2], [3, 3], [3, 2]]
        self.steps = 0
        self.total_reward = 0
        self.done = False

    def get_state(self, pos):
        return pos[0] * self.size + pos[1]

    def step(self, action_idx, q_table, alpha, gamma):
        r, c = self.robot_pos
        new_r, new_c = r, c
        
        # Mapping
        if action_idx == 0: new_r -= 1 # hoch
        elif action_idx == 1: new_r += 1 # runter
        elif action_idx == 2: new_c -= 1 # links
        elif action_idx == 3: new_c += 1 # rechts

        state = self.get_state([r, c])
        
        # 1. Kollision mit Wand
        if not (0 <= new_r < self.size and 0 <= new_c < self.size):
            reward = -15
            new_state = state
            feedback = "💥 WAND!"
        # 2. Kollision mit Hindernis
        elif [new_r, new_c] in self.obstacles:
            reward = -20
            new_state = state
            feedback = "🚫 HINDERNIS!"
        # 3. Ziel erreicht
        elif [new_r, new_c] == self.target_pos:
            reward = 100
            self.robot_pos = [new_r, new_c]
            new_state = self.get_state(self.robot_pos)
            self.done = True
            feedback = "🔋 GELADEN!"
        # 4. Normaler Schritt
        else:
            self.robot_pos = [new_r, new_c]
            new_state = self.get_state(self.robot_pos)
            # Kleine Belohnung für Richtung Ziel, Strafe für Entfernen
            old_dist = abs(r-4) + abs(c-4)
            new_dist = abs(new_r-4) + abs(new_c-4)
            reward = 2 if new_dist < old_dist else -2
            feedback = "👣 Suche..."

        # Q-Update
        old_q = q_table[state, action_idx]
        max_future_q = np.max(q_table[new_state])
        q_table[state, action_idx] = old_q + alpha * (reward + gamma * max_future_q - old_q)
        
        return reward, feedback

# --- SESSION STATE ---
if 'sim' not in st.session_state:
    st.session_state.sim = RLSimulator()
    st.session_state.q_table = np.zeros((25, 4))
    st.session_state.history = []

# --- SIDEBAR (STEUERUNG) ---
st.sidebar.header("KI Parameter")
alpha = st.sidebar.slider("Lernrate (Alpha)", 0.1, 1.0, 0.5)
gamma = st.sidebar.slider("Zukunftsglaube (Gamma)", 0.1, 1.0, 0.9)
epsilon = st.sidebar.slider("Zufallsquote (Epsilon)", 0.0, 0.5, 0.1)
speed = st.sidebar.select_slider("Geschwindigkeit", options=[0.5, 0.2, 0.05], value=0.2)

if st.sidebar.button("Gehirn löschen (Reset)"):
    st.session_state.sim.reset()
    st.session_state.q_table = np.zeros((25, 4))
    st.rerun()

# --- HAUPTBEREICH ---
st.title("🤖 Reinforcement Learning Visualisierer")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Wo ist der Roboter?")
    grid = np.full((5, 5), "⬜")
    for obs in st.session_state.sim.obstacles:
        grid[obs[0], obs[1]] = "🚫"
    grid[st.session_state.sim.target_pos[0], st.session_state.sim.target_pos[1]] = "🔋"
    grid[st.session_state.sim.robot_pos[0], st.session_state.sim.robot_pos[1]] = "🤖"
    st.table(grid)

with col2:
    st.subheader("🧠 Was weiß die KI? (Max Q-Value)")
    # Wir zeigen den maximalen Q-Wert pro Zelle als Heatmap
    q_vis = np.max(st.session_state.q_table, axis=1).reshape((5, 5))
    fig = px.imshow(q_vis, text_auto=True, color_continuous_scale='RdYlGn', 
                    labels=dict(color="Güte des Feldes"))
    fig.update_layout(width=350, height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --- TRAINING LOOP ---
if st.button("Lern-Zyklus starten"):
    status_placeholder = st.empty()
    
    for i in range(20): # 20 Schritte pro Klick
        if st.session_state.sim.done:
            st.session_state.sim.reset()
            st.session_state.sim.done = False
        
        state = st.session_state.sim.get_state(st.session_state.sim.robot_pos)
        
        # Epsilon-Greedy
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
            mode = "Zufall 🎲"
        else:
            action = np.argmax(st.session_state.q_table[state])
            mode = "Wissen 🧠"
            
        reward, fb = st.session_state.sim.step(action, st.session_state.q_table, alpha, gamma)
        st.session_state.sim.total_reward += reward
        st.session_state.sim.steps += 1
        
        status_placeholder.markdown(f"""
        <div class='status-box'>
            <b>Modus:</b> {mode} | <b>Feedback:</b> {fb} | <b>Score:</b> {st.session_state.sim.total_reward}
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(speed)
        st.rerun()

st.info("""
**Erklärung für Teilnehmer:**
- **Links (Grid):** Der Agent agiert in der echten Welt.
- **Rechts (Heatmap):** Das "Gehirn". Je grüner ein Feld, desto mehr Belohnung erwartet der Roboter dort. 
- **Lerneffekt:** Beobachte, wie nach einer Kollision (🚫) das entsprechende Feld in der Heatmap sofort 'kälter' (roter) wird!
""")
