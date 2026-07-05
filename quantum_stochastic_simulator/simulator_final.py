import numpy as np


# ============================================================
# Circuit
# ============================================================

class Circuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=np.complex128)
        self.state[0] = 1.0
        self.gates = []

    def add_gate(self, op, qubits, params_fn=None):
        self.gates.append((op, qubits, params_fn))

    def reset(self):
        self.state[:] = 0.0
        self.state[0] = 1.0

    def _apply_single_qubit(self, U, q):
        dim = 2**self.n_qubits
        mask = 1 << q
        new_state = self.state.copy()
        for i in range(dim):
            if i & mask == 0:
                j = i | mask
                a, b = self.state[i], self.state[j]
                new_state[i] = U[0, 0]*a + U[0, 1]*b
                new_state[j] = U[1, 0]*a + U[1, 1]*b
        self.state = new_state

    def apply_gate(self, op, qubits, params):
        U = op(params)
        if len(qubits) == 1:
            self._apply_single_qubit(U, qubits[0])
        else:
            raise NotImplementedError

    def run(self, theta):
        self.reset()
        for op, qubits, params_fn in self.gates:
            params = params_fn(theta) if callable(params_fn) else params_fn
            self.apply_gate(op, qubits, params)

    def expectation(self, H):
        return np.vdot(self.state, H @ self.state).real


# ============================================================
# Noise Model (SDE + RN)
# ============================================================

class NoiseModel:
    def __init__(self, a, b, a_tilde, T, dt):
        self.a = a
        self.b = b
        self.a_tilde = a_tilde
        self.T = T
        self.dt = dt

    def sample_path(self, lambda0):
        times = np.arange(0, self.T + self.dt, self.dt)
        lam = np.zeros_like(times)
        lam[0] = lambda0
        dW = np.sqrt(self.dt) * np.random.randn(len(times) - 1)
        for k in range(len(times) - 1):
            lam[k+1] = lam[k] + self.a(lam[k])*self.dt + self.b(lam[k])*dW[k]
        return times, lam, dW

    def rn_weight(self, lam, dW):
        dt = self.dt
        drift_diff = (self.a_tilde(lam[:-1]) - self.a(lam[:-1])) / self.b(lam[:-1])
        term1 = np.sum(drift_diff * dW)
        term2 = 0.5 * np.sum(drift_diff**2) * dt
        return np.exp(term1 - term2)


# ============================================================
# Shot-based Gradient Estimator
# ============================================================

class ShotGradientEstimator:
    def __init__(self, circuit, noise_model, H, shots, lambda0):
        self.circuit = circuit
        self.noise_model = noise_model
        self.H = H
        self.shots = shots
        self.lambda0 = lambda0

    def estimate_cost(self, theta):
        _, lam, dW = self.noise_model.sample_path(self.lambda0)
        w = self.noise_model.rn_weight(lam, dW)
        vals = []
        for _ in range(self.shots):
            self.circuit.run(theta)
            vals.append(self.circuit.expectation(self.H))
        return w * np.mean(vals)

    def estimate_gradient(self, theta, idx, shift):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[idx] += shift
        theta_minus[idx] -= shift
        c_plus = self.estimate_cost(theta_plus)
        c_minus = self.estimate_cost(theta_minus)
        return 0.5 * (c_plus - c_minus)


# ============================================================
# Hybrid Model + Trainer
# ============================================================

class HybridModel:
    def __init__(self, classical_pre, circuit, H_head):
        self.classical_pre = classical_pre
        self.circuit = circuit
        self.H_head = H_head

    def forward(self, x, theta):
        z = self.classical_pre(x)
        self.circuit.run(theta)
        q_feat = self.circuit.expectation(self.H_head)
        return q_feat, z


class SimpleAdam:
    def __init__(self, lr=1e-2):
        self.lr = lr
        self.m = None
        self.v = None
        self.t = 0

    def update(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = 0.9*self.m + 0.1*grad
        self.v = 0.999*self.v + 0.001*(grad**2)
        m_hat = self.m / (1 - 0.9**self.t)
        v_hat = self.v / (1 - 0.999**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class Trainer:
    def __init__(self, hybrid_model, grad_estimator, optimizer, data_loader):
        self.model = hybrid_model
        self.grad_estimator = grad_estimator
        self.optimizer = optimizer
        self.data_loader = data_loader

    def step(self, theta):
        batch_loss = 0.0
        for x, y in self.data_loader:
            q_feat, _ = self.model.forward(x, theta)
            loss = 0.5 * (q_feat - y)**2
            batch_loss += loss
        batch_loss /= len(self.data_loader)

        grads = np.zeros_like(theta)
        shift = np.pi / 2.0
        for idx in range(len(theta)):
            grads[idx] = self.grad_estimator.estimate_gradient(theta, idx, shift)

        theta_new = self.optimizer.update(theta, grads)
        return theta_new, batch_loss


# ============================================================
# Simulator Wrapper
# ============================================================

class Simulator:
    def __init__(self, circuit, noise_model, H_cost, H_head,
                 classical_pre, data_loader, lambda0,
                 shots=128, lr=1e-2):
        self.grad_estimator = ShotGradientEstimator(
            circuit=circuit,
            noise_model=noise_model,
            H=H_cost,
            shots=shots,
            lambda0=lambda0,
        )
        self.hybrid_model = HybridModel(
            classical_pre=classical_pre,
            circuit=circuit,
            H_head=H_head,
        )
        self.optimizer = SimpleAdam(lr=lr)
        self.trainer = Trainer(
            hybrid_model=self.hybrid_model,
            grad_estimator=self.grad_estimator,
            optimizer=self.optimizer,
            data_loader=data_loader,
        )

    def run(self, theta_init, n_steps):
        theta = theta_init.copy()
        history = []
        for _ in range(n_steps):
            theta, loss = self.trainer.step(theta)
            history.append(loss)
        return theta, np.array(history)


# ============================================================
# Example Usage
# ============================================================

def RX(theta):
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    return np.array([[c, -1j*s],
                     [-1j*s, c]], dtype=np.complex128)

def simple_pre(x):
    return x

def make_dummy_data(n=16):
    xs = np.linspace(-1, 1, n)
    ys = xs**2
    return list(zip(xs, ys))

if __name__ == "__main__":
    circuit = Circuit(1)
    circuit.add_gate(RX, [0], params_fn=lambda theta: theta[0])

    H = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    a = lambda lam: 0.0
    b = lambda lam: 0.1
    a_tilde = lambda lam: 0.05

    noise_model = NoiseModel(a, b, a_tilde, T=1.0, dt=0.01)
    data_loader = make_dummy_data()

    sim = Simulator(
        circuit=circuit,
        noise_model=noise_model,
        H_cost=H,
        H_head=H,
        classical_pre=simple_pre,
        data_loader=data_loader,
        lambda0=0.0,
        shots=64,
        lr=1e-2,
    )

    theta_init = np.array([0.1])
    theta_final, loss_hist = sim.run(theta_init, n_steps=20)
    print("Final θ:", theta_final)
    print("Loss history:", loss_hist)
