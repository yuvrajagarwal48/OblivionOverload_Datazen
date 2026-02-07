import React, { useMemo, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, DollarSign, BarChart3, Activity } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import { USE_MOCK } from '../config';
import * as api from '../services/api';
import './MarketDataPanel.css';

/**
 * MarketDataPanel — Market conditions and time-series charts.
 *
 * In API mode: fetches from /api/simulation/history/timeseries + /api/market/state
 * In Mock mode: reads from store (populated by mock engine)
 */
export default function MarketDataPanel() {
  const advancedMetrics = useSimulationStore((s) => s.advancedMetrics);
  const timeSeriesHistory = useSimulationStore((s) => s.timeSeriesHistory);
  const timestep = useSimulationStore((s) => s.timestep);
  const simStatus = useSimulationStore((s) => s.simStatus);
  const ingestTimeSeries = useSimulationStore((s) => s.ingestTimeSeries);
  const ingestMarketState = useSimulationStore((s) => s.ingestMarketState);
  const intervalRef = useRef(null);

  // Poll market / time-series data from API
  useEffect(() => {
    if (USE_MOCK) return;
    if (simStatus !== 'running' && simStatus !== 'paused' && simStatus !== 'done') return;

    const fetchMarket = async () => {
      try {
        const [tsRes, mktRes] = await Promise.allSettled([
          api.getTimeSeriesData('market_price,default_rate,avg_capital_ratio,liquidity_index'),
          api.getMarketState(),
        ]);
        if (tsRes.status === 'fulfilled') ingestTimeSeries(tsRes.value);
        if (mktRes.status === 'fulfilled') ingestMarketState(mktRes.value);
      } catch { /* silent */ }
    };

    fetchMarket();
    intervalRef.current = setInterval(fetchMarket, 3000);
    return () => clearInterval(intervalRef.current);
  }, [simStatus, ingestTimeSeries, ingestMarketState]);

  // Prepare chart data
  const chartData = useMemo(() => {
    const maxLen = Math.max(
      timeSeriesHistory.market_prices?.length || 0,
      timeSeriesHistory.interest_rates?.length || 0,
      timeSeriesHistory.liquidity_indices?.length || 0,
      timeSeriesHistory.default_rates?.length || 0,
      timeSeriesHistory.system_capital_ratios?.length || 0
    );

    const data = [];
    for (let i = 0; i < maxLen; i++) {
      data.push({
        step: i,
        price: timeSeriesHistory.market_prices[i] ?? null,
        rate: (timeSeriesHistory.interest_rates[i] ?? 0) * 100, // Convert to %
        liquidity: (timeSeriesHistory.liquidity_indices[i] ?? 0) * 100,
        defaultRate: (timeSeriesHistory.default_rates[i] ?? 0) * 100,
        capitalRatio: (timeSeriesHistory.system_capital_ratios[i] ?? 0) * 100,
      });
    }
    return data;
  }, [timeSeriesHistory]);

  const getRegimeColor = (regime) => {
    if (regime === 'CRISIS') return '#ef4444';
    if (regime === 'STRESSED') return '#f59e0b';
    return '#10b981';
  };

  return (
    <div className="market-data-panel">
      <div className="market-header">
        <h2 className="market-title">Market Data & Time Series</h2>
        {timestep > 0 && (
          <span className="market-step">t = {timestep}</span>
        )}
      </div>

      {/* Current Market Snapshot */}
      <div className="market-cards-grid">
        <MarketCard
          icon={DollarSign}
          label="Asset Price"
          value={advancedMetrics.market_price}
          format="currency"
          unit="₹"
        />
        <MarketCard
          icon={TrendingUp}
          label="Interest Rate"
          value={advancedMetrics.interest_rate}
          format="percent"
        />
        <MarketCard
          icon={BarChart3}
          label="Liquidity Index"
          value={advancedMetrics.liquidity_index}
          format="percent"
        />
        <MarketCard
          icon={Activity}
          label="Market Regime"
          value={advancedMetrics.market_stress_regime}
          format="text"
          color={getRegimeColor(advancedMetrics.market_stress_regime)}
        />
      </div>

      {/* Price & Rate Chart */}
      {chartData.length > 1 && (
        <div className="market-chart-section">
          <h3 className="chart-section-title">Asset Price & Interest Rate Evolution</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="step" 
                  stroke="#64748b" 
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  label={{ value: 'Timestep', position: 'insideBottom', offset: -5, fill: '#64748b' }}
                />
                <YAxis 
                  yAxisId="left"
                  stroke="#64748b" 
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  label={{ value: 'Price (₹)', angle: -90, position: 'insideLeft', fill: '#64748b' }}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  stroke="#64748b" 
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  label={{ value: 'Rate (%)', angle: 90, position: 'insideRight', fill: '#64748b' }}
                />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '6px' }}
                  labelStyle={{ color: '#94a3b8' }}
                  itemStyle={{ color: '#e2e8f0' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />
                <Line 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="price" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  dot={false}
                  name="Asset Price"
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="rate" 
                  stroke="#f59e0b" 
                  strokeWidth={2}
                  dot={false}
                  name="Interest Rate (%)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* System Health Chart */}
      {chartData.length > 1 && (
        <div className="market-chart-section">
          <h3 className="chart-section-title">System Health Indicators</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="step" 
                  stroke="#64748b" 
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  label={{ value: 'Timestep', position: 'insideBottom', offset: -5, fill: '#64748b' }}
                />
                <YAxis 
                  stroke="#64748b" 
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft', fill: '#64748b' }}
                />
                <Tooltip 
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '6px' }}
                  labelStyle={{ color: '#94a3b8' }}
                  itemStyle={{ color: '#e2e8f0' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />
                <Line 
                  type="monotone" 
                  dataKey="liquidity" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  dot={false}
                  name="Liquidity Index (%)"
                />
                <Line 
                  type="monotone" 
                  dataKey="capitalRatio" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  dot={false}
                  name="Avg Capital Ratio (%)"
                />
                <Line 
                  type="monotone" 
                  dataKey="defaultRate" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  dot={false}
                  name="Default Rate (%)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {chartData.length <= 1 && (
        <div className="market-empty">
          <p>Charts will appear as simulation progresses</p>
        </div>
      )}
    </div>
  );
}

function MarketCard({ icon: Icon, label, value, format, unit, color }) {
  let display = '—';
  
  if (value != null) {
    if (format === 'currency') {
      display = `${unit}${Number(value).toFixed(1)}`;
    } else if (format === 'percent') {
      display = `${(value * 100).toFixed(2)}%`;
    } else {
      display = String(value);
    }
  }

  return (
    <div className="market-card">
      <div className="market-card-header">
        <Icon size={16} />
        <span className="market-card-label">{label}</span>
      </div>
      <div className="market-card-value" style={color ? { color } : {}}>
        {display}
      </div>
    </div>
  );
}
