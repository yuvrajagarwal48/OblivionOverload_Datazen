import React, { useState } from 'react';
import { Zap, LogIn, AlertCircle, Building2 } from 'lucide-react';
import { signInBank, fetchBankProfile, logSessionAction } from '../lib/supabase';
import useSimulationStore from '../store/simulationStore';
import './LoginPage.css';

export default function LoginPage() {
  const setAuth = useSimulationStore((s) => s.setAuth);
  const nodes = useSimulationStore((s) => s.nodes || []);
  const backendInitialized = useSimulationStore((s) => s.backendInitialized ?? false);
  
  // Build bank list from simulation nodes (only after initialization)
  const availableBanks = backendInitialized && nodes.length > 0
    ? nodes.map((n) => ({
        bank_id: Number(n.id),
        name: n.label || `Bank ${n.id}`,
        tier: n.tier || 2,
        email: `bank${n.id}@finsim.local`,
      }))
    : [];
  
  const [mode, setMode] = useState('select'); // 'select' | 'credentials'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [selectedBank, setSelectedBank] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDemoLogin = async (bank) => {
    setLoading(true);
    setError(null);
    try {
      // For demo mode: directly authenticate with bank data
      // In production, this would use Supabase auth
      const profile = {
        bank_id: bank.bank_id,
        name: bank.name,
        tier: bank.tier,
        email: bank.email,
      };

      try {
        await logSessionAction(bank.bank_id, 'LOGIN', { mode: 'demo' });
      } catch (e) {
        // Supabase may not be configured yet, continue anyway
      }

      setAuth({
        isAuthenticated: true,
        currentBankId: bank.bank_id,
        currentBankData: profile,
        restrictedMode: true,
      });
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCredentialLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await signInBank(email, password);
      const profile = await fetchBankProfile(email);
      await logSessionAction(profile.bank_id, 'LOGIN', { mode: 'credentials' });

      setAuth({
        isAuthenticated: true,
        currentBankId: profile.bank_id,
        currentBankData: profile,
        restrictedMode: true,
      });
    } catch (err) {
      setError(err.message || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  const handleAdminLogin = () => {
    setAuth({
      isAuthenticated: true,
      currentBankId: null,
      currentBankData: null,
      restrictedMode: false,
    });
  };

  return (
    <div className="login-page">
      <div className="login-container">
        {/* Logo */}
        <div className="login-logo">
          <Zap size={28} />
          <span>FinSim<span className="login-logo-accent">-MAPPO</span></span>
        </div>

        <h1 className="login-title">Bank Portal Login</h1>
        <p className="login-subtitle">
          Select your bank to access the restricted simulation view
        </p>

        {error && (
          <div className="login-error">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Mode Toggle */}
        <div className="login-mode-toggle">
          <button
            className={`mode-btn ${mode === 'select' ? 'active' : ''}`}
            onClick={() => setMode('select')}
          >
            Quick Select
          </button>
          <button
            className={`mode-btn ${mode === 'credentials' ? 'active' : ''}`}
            onClick={() => setMode('credentials')}
          >
            Email Login
          </button>
        </div>

        {mode === 'select' ? (
          /* ── Bank Selection Grid ── */
          availableBanks.length > 0 ? (
          <div className="bank-select-grid">
            <div className="bank-tier-group">
              <h3 className="tier-group-title">
                <span className="tier-dot tier-1" />
                Tier-1 Core Banks
              </h3>
              <div className="bank-buttons">
                {availableBanks.filter((b) => b.tier === 1).map((bank) => (
                  <button
                    key={bank.bank_id}
                    className={`bank-select-btn tier-1-btn ${selectedBank?.bank_id === bank.bank_id ? 'selected' : ''}`}
                    onClick={() => setSelectedBank(bank)}
                    disabled={loading}
                  >
                    <Building2 size={16} />
                    <span>{bank.name}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="bank-tier-group">
              <h3 className="tier-group-title">
                <span className="tier-dot tier-2" />
                Tier-2 Peripheral Banks
              </h3>
              <div className="bank-buttons">
                {availableBanks.filter((b) => b.tier === 2).map((bank) => (
                  <button
                    key={bank.bank_id}
                    className={`bank-select-btn tier-2-btn ${selectedBank?.bank_id === bank.bank_id ? 'selected' : ''}`}
                    onClick={() => setSelectedBank(bank)}
                    disabled={loading}
                  >
                    <Building2 size={14} />
                    <span>{bank.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {selectedBank && (
              <button
                className="login-submit-btn"
                onClick={() => handleDemoLogin(selectedBank)}
                disabled={loading}
              >
                <LogIn size={18} />
                {loading ? 'Signing In...' : `Login as ${selectedBank.name}`}
              </button>
            )}
          </div>
          ) : (
            <div className="bank-select-empty">
              <AlertCircle size={48} style={{ color: '#64748b', marginBottom: '16px' }} />
              <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#94a3b8', marginBottom: '8px' }}>
                No Banks Available
              </h3>
              <p style={{ fontSize: '14px', color: '#64748b', textAlign: 'center', maxWidth: '320px' }}>
                Initialize the simulation first as an Observer, then return here to login as a specific bank.
              </p>
            </div>
          )
        ) : (
          /* ── Credential Form ── */
          <form className="login-form" onSubmit={handleCredentialLogin}>
            <div className="form-field">
              <label className="form-label">Email</label>
              <input
                type="email"
                className="form-input"
                placeholder="bank0@finsim.local"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="form-field">
              <label className="form-label">Password</label>
              <input
                type="password"
                className="form-input"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <button type="submit" className="login-submit-btn" disabled={loading}>
              <LogIn size={18} />
              {loading ? 'Signing In...' : 'Sign In'}
            </button>
          </form>
        )}

        {/* Admin bypass */}
        <div className="login-admin-section">
          <button className="admin-bypass-btn" onClick={handleAdminLogin}>
            Enter as Observer (Full Network View)
          </button>
        </div>
      </div>
    </div>
  );
}
