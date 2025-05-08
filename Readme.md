# 🌐 Disease Spread Modeling with Compressive Sensing

## 📖 Overview
Incomplete or delayed reporting during infectious disease outbreaks hinders accurate forecasting and timely intervention. This project integrates **Compressive Sensing (CS)** with classical epidemiological models to:

- **Reconstruct** missing case counts  
- **Identify** hidden transmission patterns  
- **Forecast** disease spread under data scarcity  
- **Scale** to big‑data settings with real‑time processing  

For detailed mathematical foundations and COVID‑19 case studies, see the project report.

---

## 🗂 Repository Structure
.
├── data_simulation.py # Simulate true/masked case data
├── transforms.py # 2D DCT / IDCT transforms
├── recovery.py # Compressive sensing recovery (LASSO)
├── network_simulation.py # Simple network‑diffusion model
├── matrix_completion.py # SoftImpute matrix completion
├── evaluation.py # RMSE & visualization utilities
├── main.py # End‑to‑end pipeline
└── requirements.txt # Python dependencies


---

## 🔬 Methodology
1. **Data Simulation & Masking**  
   - Generate Poisson‑distributed case counts  
   - Randomly mask observations to simulate under‑reporting  

2. **Sparse Representation**  
   - Apply 2D DCT to exploit spatial/temporal sparsity  

3. **Compressive Sensing Recovery**  
   - Flatten case matrix, generate Gaussian measurement matrix Φ  
   - Recover true signal via LASSO (ℓ₁‑minimization)  

4. **Network Diffusion Model**  
   - Simulate spread over an Erdős‑Rényi contact graph  

5. **Matrix Completion**  
   - Use SoftImpute for low‑rank recovery of missing entries  

6. **Evaluation & Visualization**  
   - Compute RMSE between true and recovered signals  
   - Plot side‑by‑side case matrices to assess reconstruction quality  

---

## 🚀 Getting Started
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/disease-cs.git
   cd disease-cs

2.  **Install Dependecies**  
   ```python
    pip install -r requirements.txt

3. **Run the Pipeline**  
   ```python
   python main.py
   