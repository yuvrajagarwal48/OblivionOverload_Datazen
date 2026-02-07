-- ═══════════════════════════════════════════════════════════════
-- FinSim-MAPPO Supabase Database Schema
-- ═══════════════════════════════════════════════════════════════
-- Run this in Supabase SQL Editor to create all tables + RLS policies
-- ═══════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────
-- 1. BANK PROFILES TABLE
-- ───────────────────────────────────────────────────────────────
-- Stores metadata for each bank (linked to Supabase Auth users)

CREATE TABLE IF NOT EXISTS public.bank_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  bank_id INTEGER NOT NULL UNIQUE,
  name TEXT NOT NULL,
  tier INTEGER NOT NULL CHECK (tier IN (1, 2, 3)),
  email TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_bank_profiles_bank_id ON public.bank_profiles(bank_id);
CREATE INDEX idx_bank_profiles_user_id ON public.bank_profiles(user_id);
CREATE INDEX idx_bank_profiles_email ON public.bank_profiles(email);

-- RLS Policies
ALTER TABLE public.bank_profiles ENABLE ROW LEVEL SECURITY;

-- Banks can read their own profile
CREATE POLICY "Banks can view own profile"
  ON public.bank_profiles
  FOR SELECT
  USING (auth.uid() = user_id);

-- Banks can update their own profile
CREATE POLICY "Banks can update own profile"
  ON public.bank_profiles
  FOR UPDATE
  USING (auth.uid() = user_id);

-- ───────────────────────────────────────────────────────────────
-- 2. TRANSACTIONS TABLE
-- ───────────────────────────────────────────────────────────────
-- Stores what-if simulation proposals and approvals

CREATE TABLE IF NOT EXISTS public.transactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  initiator_bank_id INTEGER NOT NULL,
  tx_type TEXT NOT NULL CHECK (tx_type IN ('LEND', 'BORROW', 'SELL_ASSETS', 'BUY_ASSETS', 'LIQUIDATE')),
  counterparty_id INTEGER,
  amount NUMERIC(15, 2) NOT NULL CHECK (amount > 0),
  outcome TEXT CHECK (outcome IN ('PASS', 'FAIL')),
  risk_before NUMERIC(5, 4),
  risk_after NUMERIC(5, 4),
  approved BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes
CREATE INDEX idx_transactions_initiator ON public.transactions(initiator_bank_id);
CREATE INDEX idx_transactions_counterparty ON public.transactions(counterparty_id);
CREATE INDEX idx_transactions_created_at ON public.transactions(created_at DESC);
CREATE INDEX idx_transactions_approved ON public.transactions(approved);

-- RLS Policies
ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;

-- Banks can view their own transactions (as initiator or counterparty)
CREATE POLICY "Banks can view own transactions"
  ON public.transactions
  FOR SELECT
  USING (
    initiator_bank_id IN (
      SELECT bank_id FROM public.bank_profiles WHERE user_id = auth.uid()
    )
    OR counterparty_id IN (
      SELECT bank_id FROM public.bank_profiles WHERE user_id = auth.uid()
    )
  );

-- Banks can insert their own transactions
CREATE POLICY "Banks can insert own transactions"
  ON public.transactions
  FOR INSERT
  WITH CHECK (
    initiator_bank_id IN (
      SELECT bank_id FROM public.bank_profiles WHERE user_id = auth.uid()
    )
  );

-- ───────────────────────────────────────────────────────────────
-- 3. SESSION LOGS TABLE
-- ───────────────────────────────────────────────────────────────
-- Tracks login/logout events and user actions for audit

CREATE TABLE IF NOT EXISTS public.session_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  bank_id INTEGER NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('LOGIN', 'LOGOUT', 'SIMULATION_START', 'SIMULATION_STOP', 'WHAT_IF_RUN', 'TRANSACTION_APPROVED')),
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes
CREATE INDEX idx_session_logs_bank_id ON public.session_logs(bank_id);
CREATE INDEX idx_session_logs_timestamp ON public.session_logs(timestamp DESC);
CREATE INDEX idx_session_logs_action ON public.session_logs(action);

-- RLS Policies
ALTER TABLE public.session_logs ENABLE ROW LEVEL SECURITY;

-- Banks can view their own logs
CREATE POLICY "Banks can view own logs"
  ON public.session_logs
  FOR SELECT
  USING (
    bank_id IN (
      SELECT bank_id FROM public.bank_profiles WHERE user_id = auth.uid()
    )
  );

-- Banks can insert their own logs
CREATE POLICY "Banks can insert own logs"
  ON public.session_logs
  FOR INSERT
  WITH CHECK (
    bank_id IN (
      SELECT bank_id FROM public.bank_profiles WHERE user_id = auth.uid()
    )
  );

-- ───────────────────────────────────────────────────────────────
-- 4. SIMULATION SNAPSHOTS TABLE (OPTIONAL)
-- ───────────────────────────────────────────────────────────────
-- Stores periodic snapshots of full simulation state for replay/analysis

CREATE TABLE IF NOT EXISTS public.simulation_snapshots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scenario TEXT NOT NULL,
  timestep INTEGER NOT NULL,
  snapshot_data JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_simulation_snapshots_scenario ON public.simulation_snapshots(scenario);
CREATE INDEX idx_simulation_snapshots_timestep ON public.simulation_snapshots(timestep);
CREATE INDEX idx_simulation_snapshots_created_at ON public.simulation_snapshots(created_at DESC);

-- RLS Policies (read-only for authenticated users)
ALTER TABLE public.simulation_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can view snapshots"
  ON public.simulation_snapshots
  FOR SELECT
  USING (auth.role() = 'authenticated');

-- Service role can insert (backend only)
CREATE POLICY "Service role can insert snapshots"
  ON public.simulation_snapshots
  FOR INSERT
  WITH CHECK (auth.role() = 'service_role');

-- ───────────────────────────────────────────────────────────────
-- 5. MARKET DATA HISTORY TABLE (OPTIONAL)
-- ───────────────────────────────────────────────────────────────
-- Stores historical market metrics for analytics

CREATE TABLE IF NOT EXISTS public.market_data_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  timestep INTEGER NOT NULL,
  market_price NUMERIC(10, 2),
  interest_rate NUMERIC(5, 4),
  liquidity_index NUMERIC(5, 4),
  default_rate NUMERIC(5, 4),
  volatility NUMERIC(5, 4),
  market_regime TEXT CHECK (market_regime IN ('NORMAL', 'STRESSED', 'CRISIS')),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_market_data_timestep ON public.market_data_history(timestep);
CREATE INDEX idx_market_data_created_at ON public.market_data_history(created_at DESC);

-- RLS Policies (read-only for authenticated users)
ALTER TABLE public.market_data_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can view market data"
  ON public.market_data_history
  FOR SELECT
  USING (auth.role() = 'authenticated');

-- ───────────────────────────────────────────────────────────────
-- 6. HELPER FUNCTIONS
-- ───────────────────────────────────────────────────────────────

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to bank_profiles
CREATE TRIGGER update_bank_profiles_updated_at
  BEFORE UPDATE ON public.bank_profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- ───────────────────────────────────────────────────────────────
-- 7. SAMPLE DATA (OPTIONAL - FOR TESTING)
-- ───────────────────────────────────────────────────────────────

-- Insert sample bank profiles (NOTE: user_id should be NULL for demo mode)
-- In production, these will be created when users sign up

INSERT INTO public.bank_profiles (bank_id, name, tier, email, user_id) VALUES
  (0, 'Core Bank 0', 1, 'bank0@finsim.local', NULL),
  (1, 'Core Bank 1', 1, 'bank1@finsim.local', NULL),
  (2, 'Core Bank 2', 1, 'bank2@finsim.local', NULL),
  (3, 'Core Bank 3', 1, 'bank3@finsim.local', NULL),
  (4, 'Core Bank 4', 1, 'bank4@finsim.local', NULL),
  (5, 'Core Bank 5', 1, 'bank5@finsim.local', NULL),
  (6, 'Peripheral Bank 6', 2, 'bank6@finsim.local', NULL),
  (7, 'Peripheral Bank 7', 2, 'bank7@finsim.local', NULL),
  (8, 'Peripheral Bank 8', 2, 'bank8@finsim.local', NULL),
  (9, 'Peripheral Bank 9', 2, 'bank9@finsim.local', NULL),
  (10, 'Peripheral Bank 10', 2, 'bank10@finsim.local', NULL),
  (11, 'Peripheral Bank 11', 2, 'bank11@finsim.local', NULL),
  (12, 'Peripheral Bank 12', 2, 'bank12@finsim.local', NULL),
  (13, 'Peripheral Bank 13', 2, 'bank13@finsim.local', NULL),
  (14, 'Peripheral Bank 14', 2, 'bank14@finsim.local', NULL),
  (15, 'Peripheral Bank 15', 2, 'bank15@finsim.local', NULL),
  (16, 'Peripheral Bank 16', 2, 'bank16@finsim.local', NULL),
  (17, 'Peripheral Bank 17', 2, 'bank17@finsim.local', NULL),
  (18, 'Peripheral Bank 18', 2, 'bank18@finsim.local', NULL),
  (19, 'Peripheral Bank 19', 2, 'bank19@finsim.local', NULL),
  (20, 'Peripheral Bank 20', 2, 'bank20@finsim.local', NULL),
  (21, 'Peripheral Bank 21', 2, 'bank21@finsim.local', NULL),
  (22, 'Peripheral Bank 22', 2, 'bank22@finsim.local', NULL),
  (23, 'Peripheral Bank 23', 2, 'bank23@finsim.local', NULL),
  (24, 'Peripheral Bank 24', 2, 'bank24@finsim.local', NULL),
  (25, 'Peripheral Bank 25', 2, 'bank25@finsim.local', NULL),
  (26, 'Peripheral Bank 26', 2, 'bank26@finsim.local', NULL),
  (27, 'Peripheral Bank 27', 2, 'bank27@finsim.local', NULL),
  (28, 'Peripheral Bank 28', 2, 'bank28@finsim.local', NULL),
  (29, 'Peripheral Bank 29', 2, 'bank29@finsim.local', NULL)
ON CONFLICT (bank_id) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════
-- DONE! Tables + RLS policies created
-- ═══════════════════════════════════════════════════════════════
-- Next steps:
-- 1. Get your Supabase project URL and anon key from dashboard
-- 2. Add to .env:
--    REACT_APP_SUPABASE_URL=https://your-project.supabase.co
--    REACT_APP_SUPABASE_ANON_KEY=your-anon-key
-- 3. For production: Create auth users and link to bank_profiles
-- 4. For demo mode: Use the sample data above (user_id = NULL)
-- ═══════════════════════════════════════════════════════════════
