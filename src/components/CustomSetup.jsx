import React, { useState, useEffect, useCallback } from 'react';
import useSimulationStore from '../store/simulationStore';
import * as api from '../services/api';
import { Search, Building2, TrendingUp, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import './CustomSetup.css';

/**
 * CustomSetup - Select real Indian banks from RBI registry + market params
 */
export default function CustomSetup() {
  const manualConfig = useSimulationStore((s) => s.manualConfig);
  const setManualConfig = useSimulationStore((s) => s.setManualConfig);
  const simStatus = useSimulationStore((s) => s.simStatus);
  const bankRegistry = useSimulationStore((s) => s.bankRegistry);
  const registryLoaded = useSimulationStore((s) => s.registryLoaded);
  const setBankRegistry = useSimulationStore((s) => s.setBankRegistry);
  const selectedRealBanks = useSimulationStore((s) => s.selectedRealBanks);
  const toggleRealBank = useSimulationStore((s) => s.toggleRealBank);
  const clearRealBankSelection = useSimulationStore((s) => s.clearRealBankSelection);

  const [activeTab, setActiveTab] = useState('banks');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const isDisabled = simStatus === 'running';

  // Load bank registry on mount
  const loadRegistry = useCallback(async () => {
    if (registryLoaded) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getBankRegistry();
      setBankRegistry(data.banks || []);
    } catch (err) {
      console.error('[Registry] Load failed:', err);
      setError('Could not load bank registry. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, [registryLoaded, setBankRegistry]);

  useEffect(() => {
    loadRegistry();
  }, [loadRegistry]);

  // Filter banks by search
  const filteredBanks = bankRegistry.filter((b) =>
    b.name?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Group by tier
  const banksByTier = filteredBanks.reduce((acc, b) => {
    const tier = b.tier || 3;
    if (!acc[tier]) acc[tier] = [];
    acc[tier].push(b);
    return acc;
  }, {});

  const tierLabels = {
    1: 'Public Sector Banks (SBI & Associates)',
    2: 'Nationalised Banks',
    3: 'Private Sector Banks',
  };

  const updateMarket = (field, value) => {
    setManualConfig('market', { ...manualConfig.market, [field]: value });
  };

  return (
    <div className="custom-setup">
      <div className="setup-header">
        <h2 className="setup-title">Bank Selection</h2>
        <p className="setup-subtitle">
          Select real Indian banks from RBI registry ({selectedRealBanks.length} selected)
        </p>
      </div>

      <div className="setup-tabs">
        <button
          className={`setup-tab ${activeTab === 'banks' ? 'active' : ''}`}
          onClick={() => setActiveTab('banks')}
          disabled={isDisabled}
        >
          <Building2 size={14} />
          <span>Banks ({selectedRealBanks.length})</span>
        </button>
        <button
          className={`setup-tab ${activeTab === 'market' ? 'active' : ''}`}
          onClick={() => setActiveTab('market')}
          disabled={isDisabled}
        >
          <TrendingUp size={14} />
          <span>Market</span>
        </button>
      </div>

      <div className="setup-content">
        {/* Banks Tab */}
        {activeTab === 'banks' && (
          <div className="setup-section">
            {/* Search + controls */}
            <div className="section-header">
              <div className="registry-search-box">
                <Search size={14} className="search-icon" />
                <input
                  type="text"
                  className="registry-search-input"
                  placeholder="Search banks..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  disabled={isDisabled}
                />
              </div>
              {selectedRealBanks.length > 0 && (
                <button
                  className="add-button"
                  onClick={clearRealBankSelection}
                  disabled={isDisabled}
                >
                  <XCircle size={14} />
                  <span>Clear All</span>
                </button>
              )}
            </div>

            {/* Loading / Error */}
            {loading && (
              <div className="empty-state">
                <Loader2 size={48} className="empty-icon spinning" />
                <p>Loading bank registry...</p>
              </div>
            )}
            {error && (
              <div className="empty-state">
                <XCircle size={48} className="empty-icon error-icon" />
                <p>{error}</p>
                <button className="add-button" onClick={loadRegistry}>
                  Retry
                </button>
              </div>
            )}

            {/* Bank list grouped by tier */}
            {!loading && !error && Object.entries(banksByTier)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([tier, banks]) => (
                <div key={tier} className="registry-tier-group">
                  <div className="registry-tier-header">
                    <span className={`tier-badge tier-${tier}`}>Tier {tier}</span>
                    <span className="registry-tier-label">
                      {tierLabels[tier] || `Tier ${tier}`} ({banks.length})
                    </span>
                  </div>
                  <div className="entity-list">
                    {banks.map((bank) => {
                      const bankId = bank.bank_id ?? bank.id;
                      const isSelected = selectedRealBanks.includes(bankId);
                      return (
                        <div
                          key={bankId}
                          className={`entity-card registry-bank-card ${isSelected ? 'selected' : ''}`}
                          onClick={() => !isDisabled && toggleRealBank(bankId)}
                        >
                          <div className="entity-header">
                            <div className="registry-bank-name">
                              {isSelected ? (
                                <CheckCircle size={16} className="check-icon selected" />
                              ) : (
                                <div className="check-placeholder" />
                              )}
                              <span className="bank-name-text">{bank.name}</span>
                            </div>
                          </div>
                          <div className="entity-fields registry-fields">
                            {bank.crar != null && (
                              <div className="field-group">
                                <label className="field-label">CRAR</label>
                                <span className="field-value">{bank.crar?.toFixed(1)}%</span>
                              </div>
                            )}
                            {bank.net_npa_ratio != null && (
                              <div className="field-group">
                                <label className="field-label">Net NPA</label>
                                <span className="field-value">{bank.net_npa_ratio?.toFixed(2)}%</span>
                              </div>
                            )}
                            {bank.total_assets != null && (
                              <div className="field-group">
                                <label className="field-label">Assets</label>
                                <span className="field-value">₹{(bank.total_assets / 1000).toFixed(0)}K Cr</span>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))
            }

            {!loading && !error && filteredBanks.length === 0 && searchQuery && (
              <div className="empty-state">
                <Search size={48} className="empty-icon" />
                <p>No banks match "{searchQuery}"</p>
              </div>
            )}
          </div>
        )}

        {/* Market Tab */}
        {activeTab === 'market' && (
          <div className="setup-section">
            <div className="section-header">
              <span className="section-label">Market Parameters</span>
            </div>

            <div className="market-params">
              <div className="param-card">
                <label className="param-label">Initial Asset Price</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.initial_price}
                  onChange={(e) => updateMarket('initial_price', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  step="0.1"
                />
                <span className="param-hint">Starting price of tradeable assets</span>
              </div>

              <div className="param-card">
                <label className="param-label">Interest Rate (%)</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.interest_rate}
                  onChange={(e) => updateMarket('interest_rate', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  max="100"
                  step="0.1"
                />
                <span className="param-hint">Base lending rate for the system</span>
              </div>

              <div className="param-card">
                <label className="param-label">Liquidity Index</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.liquidity_index}
                  onChange={(e) => updateMarket('liquidity_index', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  max="1"
                  step="0.01"
                />
                <span className="param-hint">Market liquidity measure (0-1)</span>
              </div>

              <div className="param-card">
                <label className="param-label">Volatility</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.volatility}
                  onChange={(e) => updateMarket('volatility', parseFloat(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="0"
                  max="1"
                  step="0.01"
                />
                <span className="param-hint">Price volatility factor (0-1)</span>
              </div>

              <div className="param-card">
                <label className="param-label">Episode Length</label>
                <input
                  type="number"
                  className="param-input"
                  value={manualConfig.market.episode_length}
                  onChange={(e) => updateMarket('episode_length', parseInt(e.target.value) || 0)}
                  disabled={isDisabled}
                  min="1"
                  step="1"
                />
                <span className="param-hint">Number of timesteps to simulate</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
